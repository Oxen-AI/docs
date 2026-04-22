"""Generate per-model API reference pages for the Mintlify docs site.

For each inference-capable model returned by the models API, this writes an
``inference-api/reference/models/<slug>.mdx`` page containing:

  * a short summary + workbench link
  * sample cURL and Python requests
  * a table of request parameters derived from the model's ``json_request_schema``

Run ``python generate-model-docs.py`` from the repo root. Pass ``--input
path/to/models.json`` to avoid hitting the network. Pass ``--workbench-base``
to override the default ``https://hub.oxen.ai/ai/workbench`` link.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests


DEFAULT_MODELS_URL = "https://hub.oxen.ai/api/evaluations/models"
DEFAULT_WORKBENCH_BASE = "https://www.oxen.ai/ai/workbench"
DEFAULT_OUTPUT_DIR = Path("inference-api/reference/models")

# Fallback URLs when a schema field has no `default`. Point at real files on
# hub.oxen.ai (or a plausible analogue) so the copy-paste examples work with
# minimal substitution.
FALLBACK_IMAGE_URL = (
    "https://hub.oxen.ai/api/repos/elau/assets/file/main/bloxy/bloxy_cropped_512x512.png"
)
FALLBACK_VIDEO_URL = (
    "https://hub.oxen.ai/api/repos/ox/Oxen-AI-Assets/file/main/images/winter_summer_ox.mp4"
)
# No model currently ships a default audio URL; keep a generic placeholder.
FALLBACK_AUDIO_URL = "https://example.com/audio.mp3"

SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def slugify(name: str) -> str:
    return SLUG_UNSAFE.sub("_", name).strip("_")


def load_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input:
        with open(args.input) as fh:
            payload = json.load(fh)
    else:
        response = requests.get(args.models_url, timeout=30)
        response.raise_for_status()
        payload = response.json()

    models = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        raise SystemExit(
            f"Expected a list of models; got {type(models).__name__}. "
            "If you used --input, point it at the `/api/evaluations/models` JSON payload."
        )
    return models


def pick_endpoint(model: dict[str, Any]) -> tuple[str, str]:
    """Return (endpoint_path, endpoint_type)."""
    capabilities = model.get("capabilities") or {}
    inputs = capabilities.get("input") or []
    outputs = capabilities.get("output") or []

    if "audio" in inputs and "text" in outputs and "video" not in outputs and "image" not in outputs:
        return "/api/ai/audio/transcriptions", "audio_transcribe"
    if "audio" in outputs and "video" not in outputs and "image" not in outputs:
        return "/api/ai/audio/speech", "audio_speech"
    if "video" in outputs:
        return "/api/ai/videos/generate", "video_generate"
    if "image" in outputs:
        if "image" in inputs:
            return "/api/ai/images/edit", "image_edit"
        return "/api/ai/images/generate", "image_generate"
    return "/api/ai/chat/completions", "chat"


def _x_order(field: dict[str, Any] | None) -> float:
    if not isinstance(field, dict):
        return float("inf")
    order = field.get("x-order")
    return order if isinstance(order, (int, float)) else float("inf")


def _primary_rank(name: str, required: set[str], basic: set[str]) -> int:
    return 0 if name in required or name in basic else 1


def sorted_schema_properties(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return (name, field) pairs ordered by (basic-or-required first, then advanced), then by x-order."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    basic = set(schema.get("basic") or [])
    items: list[tuple[str, dict[str, Any]]] = list(properties.items())
    items.sort(key=lambda entry: (_primary_rank(entry[0], required, basic), _x_order(entry[1])))
    return items


# Variants control how many fields each example request includes.
#   required -> minimum viable request (required fields only)
#   basic    -> required + fields the workbench surfaces by default (`basic` in the schema)
#   all      -> basic + every other field that has a schema default (strict superset of basic)
EXAMPLE_VARIANTS = ("required", "basic", "all")

EXAMPLE_VARIANT_TITLES = {
    "required": "Minimal",
    "basic": "Basic parameters",
    "all": "All parameters",
}


def example_body(model: dict[str, Any], endpoint_type: str, variant: str = "basic") -> dict[str, Any]:
    name = model.get("name", "")
    schema = (model.get("json_request_schema") or {})

    if endpoint_type == "chat":
        return _chat_example_body(model, variant)

    required = set(schema.get("required") or [])
    basic = set(schema.get("basic") or [])

    def include(field_name: str, field_schema: dict[str, Any]) -> bool:
        if variant == "required":
            return field_name in required
        if variant == "basic":
            return field_name in required or field_name in basic
        # "all" -- strict superset of basic so readers never see the "all" tab
        # missing an example input that the "basic" tab showed.
        return (
            field_name in required
            or field_name in basic
            or "default" in field_schema
        )

    body: dict[str, Any] = {"model": name}
    for field_name, field_schema in sorted_schema_properties(schema):
        if not include(field_name, field_schema):
            continue
        body[field_name] = field_placeholder(field_name, field_schema)

    if endpoint_type == "audio_transcribe" and "audio_url" not in body:
        body["audio_url"] = FALLBACK_AUDIO_URL
    if endpoint_type == "audio_speech" and "input" not in body:
        body["input"] = "Welcome to Oxen"

    return body


# Chat providers don't all accept the same parameter set. Anthropic and Google,
# for example, reject `frequency_penalty`/`presence_penalty` with a 400 error.
# Keep this keyed on provider name so per-model examples only advertise params
# the upstream provider actually accepts.
_CHAT_PARAM_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": False,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

_CHAT_PARAMS_BASIC = ("temperature", "max_tokens", "stream")
_CHAT_PARAMS_ALL_COMMON = ("temperature", "max_tokens", "stream", "top_p")

# Providers with an OpenAI-compatible interface that accept the full param set.
# Anthropic rejects `temperature` + `top_p` together with a 400, so drop `top_p`
# there (which also collapses the "All" tab into the "Basic" tab via dedup).
_CHAT_PARAMS_ALL_BY_PROVIDER = {
    "openai": _CHAT_PARAMS_ALL_COMMON + ("frequency_penalty", "presence_penalty"),
    "fireworks": _CHAT_PARAMS_ALL_COMMON + ("frequency_penalty", "presence_penalty"),
    "groq": _CHAT_PARAMS_ALL_COMMON + ("frequency_penalty", "presence_penalty"),
    "cerebras": _CHAT_PARAMS_ALL_COMMON + ("frequency_penalty", "presence_penalty"),
    "perplexity": _CHAT_PARAMS_ALL_COMMON + ("frequency_penalty", "presence_penalty"),
    "anthropic": ("temperature", "max_tokens", "stream"),
}


def _chat_param_names(model: dict[str, Any], variant: str) -> tuple[str, ...]:
    if variant == "basic":
        return _CHAT_PARAMS_BASIC
    provider = ((model.get("provider") or {}).get("name") or "").lower()
    return _CHAT_PARAMS_ALL_BY_PROVIDER.get(provider, _CHAT_PARAMS_ALL_COMMON)


def _chat_example_body(model: dict[str, Any], variant: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model.get("name", ""),
        "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    }
    if variant == "required":
        return body
    for param in _chat_param_names(model, variant):
        body[param] = _CHAT_PARAM_DEFAULTS[param]
    return body


def field_placeholder(name: str, schema_field: dict[str, Any]) -> Any:
    # Some schemas declare `default: ""` for a required string (e.g. `prompt`).
    # Using that verbatim produces an example that sends an empty prompt, which
    # the upstream provider rejects. Fall through to placeholder inference
    # instead so the example shows a usable stub like `<prompt>`.
    if "default" in schema_field and schema_field["default"] != "":
        return schema_field["default"]

    if schema_field.get("enum"):
        return schema_field["enum"][0]

    field_type = schema_field.get("type")
    render_type = schema_field.get("renderType") or ""

    if field_type == "string":
        fmt = schema_field.get("format") or ""
        if fmt == "uri" or "url" in name or render_type.endswith("-url"):
            if render_type == "audio-url" or "audio" in name:
                return FALLBACK_AUDIO_URL
            if render_type == "video-url" or "video" in name:
                return FALLBACK_VIDEO_URL
            if render_type == "image-url" or "image" in name:
                return FALLBACK_IMAGE_URL
            return FALLBACK_IMAGE_URL
        return f"<{name}>"

    if field_type == "integer":
        return schema_field.get("minimum") or 0
    if field_type == "number":
        return schema_field.get("minimum") or 0.0
    if field_type == "boolean":
        return False
    if field_type == "array":
        items_schema = schema_field.get("items") or {}
        item_render_type = items_schema.get("renderType") or render_type
        if item_render_type.endswith("-url") or any(tag in name for tag in ("image", "video", "audio")):
            return [field_placeholder(name, {"type": "string", "format": "uri", "renderType": item_render_type})]
        return [field_placeholder(f"{name}_item", {"type": items_schema.get("type", "string")})]
    if field_type == "object":
        return {}

    return f"<{name}>"


def render_schema_tables(model: dict[str, Any]) -> str:
    schema = model.get("json_request_schema") or {}
    if not (schema.get("properties") or {}):
        return ""

    required = set(schema.get("required") or [])
    ordered = sorted_schema_properties(schema)
    required_rows = [entry for entry in ordered if entry[0] in required]
    optional_rows = [entry for entry in ordered if entry[0] not in required]

    sections: list[str] = []
    if required_rows:
        sections.append("### Required parameters\n\n" + _render_parameter_table(required_rows))
    if optional_rows:
        sections.append("### Optional parameters\n\n" + _render_parameter_table(optional_rows))
    return "\n\n".join(sections)


def _render_parameter_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Field | Type | Default | Description |",
        "|-------|------|---------|-------------|",
    ]

    for prop_name, prop_schema in rows:
        field_type = prop_schema.get("type", "any")
        if field_type == "array":
            items_type = (prop_schema.get("items") or {}).get("type", "any")
            field_type = f"array<{items_type}>"

        default = prop_schema.get("default")
        default_cell = f"`{json.dumps(default)}`" if default is not None else "&mdash;"

        description = prop_schema.get("description") or prop_schema.get("title") or ""
        if prop_schema.get("enum"):
            description = (description + f" One of: {', '.join(str(v) for v in prop_schema['enum'])}.").strip()

        minimum = prop_schema.get("minimum")
        maximum = prop_schema.get("maximum")
        if minimum is not None or maximum is not None:
            description = (description + f" Range: {minimum if minimum is not None else '-∞'} – {maximum if maximum is not None else '+∞'}.").strip()

        fmt = prop_schema.get("format")
        if fmt:
            description = (description + f" Format: {fmt}.").strip()

        lines.append(
            f"| `{prop_name}` | `{field_type}` | {default_cell} | {description.strip() or ' '} |"
        )

    return "\n".join(lines)


def _shell_escape_single_quoted(text: str) -> str:
    """Make `text` safe to embed inside a `'...'` bash literal.

    Replaces each `'` with `'\\''`: close the surrounding quote, include a
    backslash-escaped apostrophe, then reopen the surrounding quote. Without this,
    a prompt containing an apostrophe aborts the shell parse with `unexpected EOF`.
    """
    return text.replace("'", "'\\''")


def render_curl(endpoint: str, body: dict[str, Any]) -> str:
    pretty = _shell_escape_single_quoted(json.dumps(body, indent=2))
    return (
        f"curl -X POST https://hub.oxen.ai{endpoint} \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -H \"Authorization: Bearer $OXEN_API_KEY\" \\\n"
        f"  -d '{pretty}'"
    )


def render_python(endpoint: str, body: dict[str, Any]) -> str:
    pretty = json.dumps(body, indent=4)
    # Indent the JSON so it nests cleanly inside the requests.post call.
    indented = "\n".join(
        line if i == 0 else f"    {line}"
        for i, line in enumerate(pretty.splitlines())
    )
    return (
        "import os\n"
        "import requests\n\n"
        "response = requests.post(\n"
        f"    \"https://hub.oxen.ai{endpoint}\",\n"
        "    headers={\n"
        "        \"Content-Type\": \"application/json\",\n"
        "        \"Authorization\": f\"Bearer {os.environ['OXEN_API_KEY']}\",\n"
        "    },\n"
        f"    json={indented},\n"
        ")\n"
        "response.raise_for_status()\n"
        "print(response.json())"
    )


ASYNC_ENDPOINT_TYPES = {"image_generate", "image_edit", "video_generate"}


def _json_body_py(body: dict[str, Any], *, indent_prefix: str = "    ") -> str:
    """Render the body as a Python dict literal, indented to nest cleanly inside a call."""
    pretty = json.dumps(body, indent=4)
    return "\n".join(
        line if i == 0 else f"{indent_prefix}{line}"
        for i, line in enumerate(pretty.splitlines())
    )


def render_async_poll_curl(body: dict[str, Any]) -> str:
    """Enqueue the job, capture the generation id, and poll that id until it 404s.

    The queue endpoint drops finished generations, so a 404 means "done"; the
    actual result URL is only emitted over SSE (see the SSE variant) or persisted
    to the caller's namespace.
    """
    pretty = _shell_escape_single_quoted(json.dumps(body, indent=2))
    return (
        "# Enqueue, capture the generation id.\n"
        "GEN_ID=$(curl -s -X POST https://hub.oxen.ai/api/ai/queue \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -H \"Authorization: Bearer $OXEN_API_KEY\" \\\n"
        f"  -d '{pretty}' | jq -r '.generations[0].generation_id')\n"
        "\n"
        "# Poll the single generation until it 404s (terminal state).\n"
        "while curl -s -o /dev/null -w \"%{http_code}\" \\\n"
        "    -H \"Authorization: Bearer $OXEN_API_KEY\" \\\n"
        "    \"https://hub.oxen.ai/api/ai/queue/$GEN_ID\" | grep -q \"^200$\"; do\n"
        "  sleep 5\n"
        "done\n"
        "echo \"Done. See the 'Async with SSE' tab to receive the result URL.\""
    )


def render_async_poll_python(body: dict[str, Any]) -> str:
    indented = _json_body_py(body)
    return (
        "import os\n"
        "import time\n"
        "import requests\n"
        "\n"
        "HEADERS = {\n"
        "    \"Content-Type\": \"application/json\",\n"
        "    \"Authorization\": f\"Bearer {os.environ['OXEN_API_KEY']}\",\n"
        "}\n"
        "\n"
        "enqueue = requests.post(\n"
        "    \"https://hub.oxen.ai/api/ai/queue\",\n"
        "    headers=HEADERS,\n"
        f"    json={indented},\n"
        ")\n"
        "enqueue.raise_for_status()\n"
        "generation_id = enqueue.json()[\"generations\"][0][\"generation_id\"]\n"
        "\n"
        "while True:\n"
        "    resp = requests.get(\n"
        "        f\"https://hub.oxen.ai/api/ai/queue/{generation_id}\",\n"
        "        headers=HEADERS,\n"
        "    )\n"
        "    if resp.status_code == 404:\n"
        "        break\n"
        "    time.sleep(5)\n"
        "print(\"Done. See the 'Async with SSE' tab to receive the result URL.\")"
    )


def render_async_sse_curl(body: dict[str, Any]) -> str:
    """Enqueue, then stream GET /api/events and print the matching completion payload.

    SSE framing (`event: ...\\ndata: ...\\n\\n`) isn't JSON-lines, so we need a
    small awk preprocessor to pull the `data:` payload that follows the
    `media_generation_completed` event before handing it to jq for filtering.
    """
    pretty = _shell_escape_single_quoted(json.dumps(body, indent=2))
    return (
        "# Enqueue, capture the generation id.\n"
        "GEN_ID=$(curl -s -X POST https://hub.oxen.ai/api/ai/queue \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -H \"Authorization: Bearer $OXEN_API_KEY\" \\\n"
        f"  -d '{pretty}' | jq -r '.generations[0].generation_id')\n"
        "\n"
        "# Stream the SSE channel, grab the data line that follows a\n"
        "# media_generation_completed event for our id, and pretty-print it.\n"
        "curl -sN -H \"Authorization: Bearer $OXEN_API_KEY\" https://hub.oxen.ai/api/events \\\n"
        "  | awk -v id=\"$GEN_ID\" '\n"
        "    /^event: media_generation_completed$/ { expect=1; next }\n"
        "    /^data: / && expect {\n"
        "      payload = substr($0, 7)\n"
        "      if (index(payload, \"\\\"generation_id\\\":\\\"\" id \"\\\"\")) { print payload; exit }\n"
        "      expect = 0\n"
        "    }\n"
        "  ' | jq ."
    )


def render_async_sse_python(body: dict[str, Any]) -> str:
    indented = _json_body_py(body)
    return (
        "import json\n"
        "import os\n"
        "import requests\n"
        "\n"
        "API_KEY = os.environ[\"OXEN_API_KEY\"]\n"
        "AUTH = {\"Authorization\": f\"Bearer {API_KEY}\"}\n"
        "\n"
        "enqueue = requests.post(\n"
        "    \"https://hub.oxen.ai/api/ai/queue\",\n"
        "    headers={**AUTH, \"Content-Type\": \"application/json\"},\n"
        f"    json={indented},\n"
        ")\n"
        "enqueue.raise_for_status()\n"
        "generation_id = enqueue.json()[\"generations\"][0][\"generation_id\"]\n"
        "\n"
        "with requests.get(\n"
        "    \"https://hub.oxen.ai/api/events\",\n"
        "    headers=AUTH,\n"
        "    stream=True,\n"
        ") as stream:\n"
        "    event_name = None\n"
        "    for line in stream.iter_lines(decode_unicode=True):\n"
        "        if line.startswith(\"event: \"):\n"
        "            event_name = line.removeprefix(\"event: \")\n"
        "        elif line.startswith(\"data: \") and event_name == \"media_generation_completed\":\n"
        "            payload = json.loads(line.removeprefix(\"data: \"))\n"
        "            if payload.get(\"generation_id\") == generation_id:\n"
        "                print(payload)\n"
        "                break"
    )


def _frontmatter_description(model: dict[str, Any]) -> str:
    """A single clean sentence to render as the page subtitle.

    The `description` field can be multiple paragraphs, which gets awkwardly
    truncated mid-word by Mintlify. We prefer the short `summary` field if the
    model provides one, then fall back to the first sentence of the description.
    """
    summary = (model.get("summary") or "").strip()
    if summary:
        return _one_line(summary, limit=160)

    description = (model.get("description") or "").strip()
    if not description:
        return "Auto-generated API reference."

    first_sentence = description.split(". ")[0].strip()
    if len(first_sentence) < len(description) and not first_sentence.endswith("."):
        first_sentence += "."
    return _one_line(first_sentence, limit=160)


def _one_line(text: str, *, limit: int) -> str:
    """Collapse whitespace and cap length at a word boundary."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    truncated = collapsed[: limit - 1].rsplit(" ", 1)[0]
    return truncated.rstrip(",;:-") + "…"


def _yaml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def _indent_lines(lines: list[str], prefix: str) -> list[str]:
    """Indent every logical line in `lines`, including ones that are themselves multiline blocks."""
    return [_indent(line, prefix) for line in lines]


# Where each endpoint's top-level reference doc lives, for the per-tab
# "See the X reference for more details" link.
_ENDPOINT_REFERENCE_DOCS = {
    "chat": ("chat completions reference", "/inference-api/reference/chat_completions"),
    "image_generate": ("image generation reference", "/inference-api/reference/image_generation"),
    "image_edit": ("image editing reference", "/inference-api/reference/image_editing"),
    "video_generate": ("video generation reference", "/inference-api/reference/video_generation"),
}

_ASYNC_QUEUE_DOC = ("async queue reference", "/inference-api/reference/async_queue")


def _render_example_block(
    variants: list[tuple[str, dict[str, Any]]],
    render_bash: Any,
    render_python_fn: Any,
) -> list[str]:
    """Render a CodeGroup (single variant) or Tabs (multiple variants) using the given renderers.

    Callers pass different renderers for sync vs. async so the same Minimal/Basic/All tab
    structure can be reused for both.
    """
    if len(variants) == 1:
        _, only_body = variants[0]
        return [
            "<CodeGroup>",
            "",
            "```bash cURL",
            render_bash(only_body),
            "```",
            "",
            "```python Python",
            render_python_fn(only_body),
            "```",
            "",
            "</CodeGroup>",
        ]

    lines: list[str] = ["<Tabs>"]
    for variant, variant_body in variants:
        lines += [
            f'  <Tab title="{EXAMPLE_VARIANT_TITLES[variant]}">',
            "",
            "    <CodeGroup>",
            "",
            "    ```bash cURL",
            _indent(render_bash(variant_body), "    "),
            "    ```",
            "",
            "    ```python Python",
            _indent(render_python_fn(variant_body), "    "),
            "    ```",
            "",
            "    </CodeGroup>",
            "",
            "  </Tab>",
        ]
    lines.append("</Tabs>")
    return lines


def render_models_endpoint_curl(model_name: str) -> str:
    return (
        "```bash\n"
        f'curl -H "Authorization: Bearer $OXEN_API_KEY" https://hub.oxen.ai/api/ai/models/{model_name}\n'
        "```"
    )


def _capability_pills(modalities: list[str]) -> str:
    if not modalities:
        return "&mdash;"
    # Render each modality as an inline code span so it reads as a pill/chip.
    return " ".join(f"`{m}`" for m in modalities)


def _render_capabilities_line(inputs: list[str], outputs: list[str]) -> str:
    """One-line flow: inputs → outputs, rendered as code-span pills."""
    return f"{_capability_pills(inputs)} &rarr; {_capability_pills(outputs)}"


def _reference_link_md(entry: tuple[str, str]) -> str:
    title, href = entry
    return f"See the [{title}]({href}) for more details."


def _wrap_in_tab(title: str, header_paragraphs: list[str], inner_block: list[str]) -> list[str]:
    """Put `header_paragraphs` + the indented inner block under a single `<Tab>`.

    Each header_paragraph is rendered on its own line with a blank line between it
    and the next paragraph (and between it and the inner block), so adjacent
    paragraphs render as distinct paragraphs rather than wrapping into one.
    """
    lines = [f'  <Tab title="{title}">', ""]
    for i, paragraph in enumerate(header_paragraphs):
        if i > 0:
            lines.append("")
        lines.append(f"    {paragraph}" if paragraph else "")
    if header_paragraphs:
        lines.append("")
    lines += _indent_lines(inner_block, "    ")
    lines += ["", "  </Tab>"]
    return lines


def _render_sync_async_sse_tabs(
    endpoint: str,
    endpoint_type: str,
    kept_variants: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Render a Sync / Async / Async-with-SSE outer Tabs block.

    Each outer tab nests the existing Minimal/Basic/All variant block, so readers
    pick their execution mode first and their parameter depth second.
    """
    sync_header: list[str] = []
    if endpoint_type == "video_generate":
        sync_header.append(
            "This blocks until the video is ready (typically 5-15 minutes)."
            " Prefer **Async** or **Async with SSE** for anything beyond quick experimentation."
        )
    sync_header.append(_reference_link_md(_ENDPOINT_REFERENCE_DOCS[endpoint_type]))

    sync_block = _render_example_block(
        kept_variants,
        lambda body: render_curl(endpoint, body),
        lambda body: render_python(endpoint, body),
    )
    async_poll_block = _render_example_block(
        kept_variants,
        lambda body: render_async_poll_curl(body),
        lambda body: render_async_poll_python(body),
    )
    async_sse_block = _render_example_block(
        kept_variants,
        lambda body: render_async_sse_curl(body),
        lambda body: render_async_sse_python(body),
    )

    async_link = [_reference_link_md(_ASYNC_QUEUE_DOC)]

    lines = ["<Tabs>"]
    lines += _wrap_in_tab("Sync", sync_header, sync_block)
    lines += _wrap_in_tab("Async", async_link, async_poll_block)
    lines += _wrap_in_tab("Async with SSE", async_link, async_sse_block)
    lines.append("</Tabs>")
    return lines


def render_page(model: dict[str, Any], workbench_base: str) -> str:
    name = model.get("name") or ""
    display_name = model.get("display_name") or name
    description = (model.get("description") or "").strip()
    endpoint, endpoint_type = pick_endpoint(model)
    workbench_url = f"{workbench_base}?model={quote(name, safe='')}"
    capabilities = model.get("capabilities") or {}
    inputs = capabilities.get("input") or []
    outputs = capabilities.get("output") or []

    front_matter = (
        "---\n"
        f'title: "{_yaml_escape(display_name)}"\n'
        f'description: "{_yaml_escape(_frontmatter_description(model))}"\n'
        "---\n"
    )

    # Render the model id in a fenced code block so Mintlify surfaces a copy
    # button on it. The display name is already the page title.
    body_md = [
        front_matter,
        "",
        "```",
        name,
        "```",
        "",
        _render_capabilities_line(inputs, outputs),
        "",
        "<CardGroup cols={1}>",
        f'  <Card title="Try {display_name} in the Workbench" icon="flask" href="{workbench_url}">',
        "    Run this model interactively, tune parameters, and compare outputs.",
        "  </Card>",
        "</CardGroup>",
        "",
        "<Tip>",
        f"  Use the [Workbench]({workbench_url}) as a request builder: configure parameters"
        " for this model in the UI, then open the **API** tab to copy the exact cURL or"
        " Python call.",
        "</Tip>",
    ]

    if description:
        body_md += ["", description]

    # Drop any variant whose body is identical to the previous kept variant so we
    # don't render duplicate tabs (e.g. a model with no `basic` fields would otherwise
    # show identical Required and Basic tabs).
    kept_variants: list[tuple[str, dict[str, Any]]] = []
    for variant in EXAMPLE_VARIANTS:
        variant_body = example_body(model, endpoint_type, variant)
        if kept_variants and variant_body == kept_variants[-1][1]:
            continue
        kept_variants.append((variant, variant_body))

    body_md += ["", "## Example request", ""]
    if endpoint_type in ASYNC_ENDPOINT_TYPES:
        body_md += _render_sync_async_sse_tabs(endpoint, endpoint_type, kept_variants)
    else:
        body_md += _render_example_block(
            kept_variants,
            lambda body: render_curl(endpoint, body),
            lambda body: render_python(endpoint, body),
        )

    body_md += [
        "",
        "## Fetch model details",
        "",
        "The [models endpoint](/inference-api/reference/models/overview) returns the full model"
        " object, including its `json_request_schema`.",
        "",
        render_models_endpoint_curl(name),
    ]

    schema_tables = render_schema_tables(model)
    if schema_tables:
        body_md += [
            "",
            "## Request parameters",
            "",
            schema_tables,
        ]
    else:
        body_md += [
            "",
            "## Request parameters",
            "",
            "This model follows the standard OpenAI chat completions request body. See the"
            " [chat completions reference](../inference-api.mdx) for the full parameter list.",
        ]

    body_md += [""]

    return "\n".join(body_md)


def iter_inference_models(models: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for model in models:
        if not model.get("can_run_inference", True):
            continue
        if model.get("deprecated"):
            continue
        if not model.get("name"):
            continue
        # Only publish references for base models. Custom (user-deployed) models
        # can be listed on hub.oxen.ai but aren't callable for everyone, so their
        # generated pages 404 for anyone else.
        if model.get("model_type") != "base":
            continue
        # Embedding models have no public chat/image/video endpoint in the
        # inference API; pick_endpoint would misroute them to /chat/completions
        # and the generated example would return 404 from the upstream provider.
        outputs = (model.get("capabilities") or {}).get("output") or []
        if "embeddings" in outputs:
            continue
        yield model


def write_pages(models: list[dict[str, Any]], output_dir: Path, workbench_base: str) -> list[tuple[str, dict[str, Any]]]:
    """Write one .mdx per model and return (file_path, model) pairs for nav building."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, dict[str, Any]]] = []

    for model in iter_inference_models(models):
        slug = slugify(model["name"])
        path = output_dir / f"{slug}.mdx"
        path.write_text(render_page(model, workbench_base))
        written.append((str(path), model))
    return written


def _developer_label(model: dict[str, Any]) -> str:
    """Pick a display label to bucket a model under on the index page."""
    dev = model.get("developer")
    if isinstance(dev, dict):
        name = dev.get("display_name") or dev.get("name")
        if name:
            return str(name)
    provider = model.get("provider")
    if isinstance(provider, dict):
        name = provider.get("display_name") or provider.get("name")
        if name:
            return str(name)
    return "Other"


def render_index_page(
    generated: list[tuple[str, dict[str, Any]]],
    nav_prefix: str,
) -> str:
    """Render a single index page that links to every generated model reference.

    Each developer gets a subsection. One page in the sidebar avoids Mintlify's
    eager render of a flat 130-leaf nav under an expanded tab.
    """
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for path, model in generated:
        buckets.setdefault(_developer_label(model), []).append((path, model))

    lines = [
        "---",
        'title: "Model API References"',
        'description: "Browse the full list of models available in the Oxen.AI inference API."',
        "---",
        "",
        "Every model exposed by the Oxen.AI inference API has a dedicated reference page"
        " with a sample request, Python snippet, and the full request schema. Models are"
        " grouped below by developer.",
        "",
        "<CardGroup cols={2}>",
        '  <Card title="Models API endpoint" icon="code" href="/inference-api/reference/models/overview">',
        "    List, search, and fetch model details programmatically via `GET /api/ai/models`.",
        "  </Card>",
        '  <Card title="Discover models on our Models page" icon="compass" href="https://www.oxen.ai/ai/models">',
        "    Filter by modality, search by name, and preview every model side-by-side before diving into a reference page.",
        "  </Card>",
        "</CardGroup>",
        "",
    ]

    for label in sorted(buckets.keys(), key=lambda l: l.lower()):
        lines.append(f"## {label}")
        lines.append("")
        for path, model in sorted(buckets[label], key=lambda entry: entry[1].get("name", "")):
            slug = Path(path).stem
            display_name = model.get("display_name") or model.get("name")
            lines.append(f"- [{display_name}](/{nav_prefix}/{slug})")
        lines.append("")

    return "\n".join(lines)


def write_index_page(
    generated: list[tuple[str, dict[str, Any]]],
    index_path: Path,
    nav_prefix: str,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(render_index_page(generated, nav_prefix))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="Path to a JSON file containing the models list (skip the network request).",
    )
    parser.add_argument(
        "--models-url",
        default=DEFAULT_MODELS_URL,
        help=f"Source URL for the models list (default: {DEFAULT_MODELS_URL}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for generated .mdx files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--workbench-base",
        default=DEFAULT_WORKBENCH_BASE,
        help=f"Base URL for workbench links (default: {DEFAULT_WORKBENCH_BASE}).",
    )
    parser.add_argument(
        "--index-path",
        default="inference-api/reference/model-references.mdx",
        help="Path to the generated index page that links to every model reference.",
    )
    parser.add_argument(
        "--nav-prefix",
        default="inference-api/reference/models",
        help="Doc path prefix used when linking to generated pages from the index.",
    )
    args = parser.parse_args()

    models = load_models(args)
    output_dir = Path(args.output)
    written = write_pages(models, output_dir, args.workbench_base)

    print(f"Wrote {len(written)} model reference page(s) under {output_dir}", file=sys.stderr)
    for path, _ in written:
        print(path)

    if args.index_path:
        index_path = Path(args.index_path)
        write_index_page(written, index_path, args.nav_prefix)
        print(f"Wrote index page {index_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
