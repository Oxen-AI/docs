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
#   all      -> required + every field that has a schema default
EXAMPLE_VARIANTS = ("required", "basic", "all")

EXAMPLE_VARIANT_TITLES = {
    "required": "Minimal",
    "basic": "Basic parameters",
    "all": "All parameters with defaults",
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
        # "all"
        return field_name in required or "default" in field_schema

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


def render_async_curl(body: dict[str, Any], model_name: str) -> str:
    pretty = _shell_escape_single_quoted(json.dumps(body, indent=2))
    quoted_name = quote(model_name, safe="")
    return (
        "# Enqueue\n"
        "curl -X POST https://hub.oxen.ai/api/ai/queue \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -H \"Authorization: Bearer $OXEN_API_KEY\" \\\n"
        f"  -d '{pretty}'\n"
        "\n"
        "# Poll until the generation drops off the list\n"
        "curl -H \"Authorization: Bearer $OXEN_API_KEY\" \\\n"
        f"  \"https://hub.oxen.ai/api/ai/queue?model={quoted_name}\""
    )


def render_async_python(body: dict[str, Any], model_name: str) -> str:
    pretty = json.dumps(body, indent=4)
    indented = "\n".join(
        line if i == 0 else f"    {line}"
        for i, line in enumerate(pretty.splitlines())
    )
    escaped_name = model_name.replace('"', '\\"')
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
        "# Enqueue\n"
        "response = requests.post(\n"
        "    \"https://hub.oxen.ai/api/ai/queue\",\n"
        "    headers=HEADERS,\n"
        f"    json={indented},\n"
        ")\n"
        "generations = response.json()[\"generations\"]\n"
        "print(f\"Enqueued {len(generations)} generation(s)\")\n"
        "\n"
        "# Poll until done\n"
        "while True:\n"
        "    status = requests.get(\n"
        "        \"https://hub.oxen.ai/api/ai/queue\",\n"
        "        headers=HEADERS,\n"
        f"        params={{\"model\": \"{escaped_name}\"}},\n"
        "    ).json()\n"
        "    if status[\"count\"] == 0:\n"
        "        break\n"
        "    time.sleep(10)\n"
        "\n"
        "print(\"Done!\")"
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


def render_page(model: dict[str, Any], workbench_base: str) -> str:
    name = model.get("name") or ""
    display_name = model.get("display_name") or name
    description = (model.get("description") or "").strip()
    endpoint, endpoint_type = pick_endpoint(model)
    workbench_url = f"{workbench_base}?model={quote(name, safe='')}"
    capabilities = model.get("capabilities") or {}
    inputs = ", ".join(capabilities.get("input") or []) or "—"
    outputs = ", ".join(capabilities.get("output") or []) or "—"

    front_matter = (
        "---\n"
        f'title: "{_yaml_escape(display_name)}"\n'
        f'description: "{_yaml_escape(_frontmatter_description(model))}"\n'
        "---\n"
    )

    body_md = [
        front_matter,
        "",
        "<CardGroup cols={1}>",
        f'  <Card title="Try {display_name} in the Workbench" icon="flask" href="{workbench_url}">',
        "    Run this model interactively, tune parameters, and compare outputs.",
        "  </Card>",
        "</CardGroup>",
        "",
        f"**Model ID:** `{name}`  ",
        f"**Endpoint:** `POST https://hub.oxen.ai{endpoint}`  ",
        f"**Inputs:** {inputs}  ",
        f"**Outputs:** {outputs}",
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
    body_md += _render_example_block(
        kept_variants,
        lambda body: render_curl(endpoint, body),
        lambda body: render_python(endpoint, body),
    )

    if endpoint_type in ASYNC_ENDPOINT_TYPES:
        body_md += [
            "",
            "## Async example",
            "",
            "Enqueue the same body and poll the queue instead of waiting synchronously. The async"
            " queue avoids long-lived HTTP connections and lets you run up to 4 generations in"
            " parallel. See the [Async Queue quick start](/inference-api/quickstart/async-queue)"
            " for the full workflow.",
            "",
        ]
        body_md += _render_example_block(
            kept_variants,
            lambda body: render_async_curl(body, name),
            lambda body: render_async_python(body, name),
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
