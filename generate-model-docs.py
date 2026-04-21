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

import requests


DEFAULT_MODELS_URL = "https://hub.oxen.ai/api/evaluations/models"
DEFAULT_WORKBENCH_BASE = "https://hub.oxen.ai/ai/workbench"
DEFAULT_OUTPUT_DIR = Path("inference-api/reference/models")

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


def example_body(model: dict[str, Any], endpoint_type: str) -> dict[str, Any]:
    name = model.get("name", "")
    schema = (model.get("json_request_schema") or {})
    properties = schema.get("properties") or {}

    if endpoint_type == "chat":
        return {
            "model": name,
            "messages": [{"role": "user", "content": "Hello, what can you do?"}],
        }

    body: dict[str, Any] = {"model": name}
    required_fields = sorted(
        schema.get("required") or [],
        key=lambda f: _x_order(properties.get(f)),
    )

    for field in required_fields:
        schema_field = properties.get(field, {})
        body[field] = field_placeholder(field, schema_field)

    # Include `prompt` by default even when not strictly required — it's the most
    # common parameter and users recognize it.
    if "prompt" in properties and "prompt" not in body:
        body["prompt"] = field_placeholder("prompt", properties["prompt"])

    if endpoint_type == "audio_transcribe" and "audio_url" not in body:
        body["audio_url"] = "https://example.com/recording.mp3"
    if endpoint_type == "audio_speech" and "input" not in body:
        body["input"] = "Welcome to Oxen"

    return body


def field_placeholder(name: str, schema_field: dict[str, Any]) -> Any:
    if "default" in schema_field:
        return schema_field["default"]

    if schema_field.get("enum"):
        return schema_field["enum"][0]

    field_type = schema_field.get("type")

    if field_type == "string":
        fmt = schema_field.get("format") or ""
        if fmt == "uri" or "url" in name:
            if "audio" in name:
                return "https://example.com/recording.mp3"
            if "video" in name:
                return "https://example.com/video.mp4"
            if "image" in name:
                return "https://example.com/image.png"
            return "https://example.com/input.bin"
        return f"<{name}>"

    if field_type == "integer":
        return schema_field.get("minimum") or 0
    if field_type == "number":
        return schema_field.get("minimum") or 0.0
    if field_type == "boolean":
        return False
    if field_type == "array":
        item_type = (schema_field.get("items") or {}).get("type", "string")
        return [field_placeholder(f"{name}_item", {"type": item_type})]
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


def render_curl(endpoint: str, body: dict[str, Any]) -> str:
    pretty = json.dumps(body, indent=2)
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


def render_page(model: dict[str, Any], workbench_base: str) -> str:
    name = model.get("name") or ""
    display_name = model.get("display_name") or name
    description = (model.get("description") or model.get("summary") or "").strip()
    endpoint, endpoint_type = pick_endpoint(model)
    body = example_body(model, endpoint_type)
    workbench_url = f"{workbench_base}?model={slugify(name)}"
    capabilities = model.get("capabilities") or {}
    inputs = ", ".join(capabilities.get("input") or []) or "—"
    outputs = ", ".join(capabilities.get("output") or []) or "—"

    front_matter = (
        "---\n"
        f"title: \"{display_name}\"\n"
        f"description: \"{description[:150] if description else 'Auto-generated API reference.'}\"\n"
        "---\n"
    )

    body_md = [
        front_matter,
        "",
        f"[Try **{display_name}** in the Workbench]({workbench_url}) &rarr;",
        "",
        f"**Model ID:** `{name}`  ",
        f"**Endpoint:** `POST https://hub.oxen.ai{endpoint}`  ",
        f"**Inputs:** {inputs}  ",
        f"**Outputs:** {outputs}",
    ]

    if description:
        body_md += ["", description]

    body_md += [
        "",
        "## Example request",
        "",
        "<CodeGroup>",
        "",
        "```bash cURL",
        render_curl(endpoint, body),
        "```",
        "",
        "```python Python",
        render_python(endpoint, body),
        "```",
        "",
        "</CodeGroup>",
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

    body_md += [
        "",
        "To fetch this schema programmatically, see"
        f" [the models overview](./overview) and call `GET /api/ai/models/{name}`.",
        "",
    ]

    return "\n".join(body_md)


def iter_inference_models(models: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for model in models:
        if not model.get("can_run_inference", True):
            continue
        if model.get("deprecated"):
            continue
        if not model.get("name"):
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
    """Extract a display label to group models under in the nav."""
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


def build_nav_groups(
    generated: list[tuple[str, dict[str, Any]]], nav_prefix: str
) -> list[dict[str, Any]]:
    """Build a list of {group, pages} nav entries, one per developer, alphabetically sorted.

    Grouping keeps each Mintlify nav group small so the sidebar stays responsive on
    expand, instead of rendering ~130 leaves in one flat list.
    """
    buckets: dict[str, list[str]] = {}
    for path, model in generated:
        label = _developer_label(model)
        page_id = f"{nav_prefix}/{Path(path).stem}"
        buckets.setdefault(label, []).append(page_id)

    return [
        {"group": label, "pages": sorted(set(pages))}
        for label, pages in sorted(buckets.items(), key=lambda entry: entry[0].lower())
    ]


def update_mint_nav(
    mint_path: Path,
    nav_groups: list[dict[str, Any]],
) -> bool:
    """Rewrite the ``Model References`` group in mint.json to contain the given sub-groups.

    Returns ``True`` if the file changed on disk, ``False`` otherwise.
    """
    if not mint_path.exists():
        return False

    with mint_path.open() as fh:
        config = json.load(fh)

    changed = False

    def walk(node: Any) -> Any:
        nonlocal changed
        if isinstance(node, dict):
            if node.get("group") == "Model References" and "pages" in node:
                if node["pages"] != nav_groups:
                    node["pages"] = nav_groups
                    changed = True
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    updated = walk(config)

    if changed:
        with mint_path.open("w") as fh:
            json.dump(updated, fh, indent=4)
            fh.write("\n")

    return changed


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
        "--mint-config",
        default="mint.json",
        help="Path to the Mintlify config file. Use '' to skip nav update.",
    )
    parser.add_argument(
        "--nav-prefix",
        default="inference-api/reference/models",
        help="Prefix used when registering generated pages in mint.json.",
    )
    args = parser.parse_args()

    models = load_models(args)
    output_dir = Path(args.output)
    written = write_pages(models, output_dir, args.workbench_base)

    print(f"Wrote {len(written)} model reference page(s) under {output_dir}", file=sys.stderr)
    for path, _ in written:
        print(path)

    if args.mint_config:
        nav_groups = build_nav_groups(written, args.nav_prefix)
        mint_changed = update_mint_nav(Path(args.mint_config), nav_groups)
        print(
            f"mint.json {'updated' if mint_changed else 'already in sync'} "
            f"({len(nav_groups)} developer groups)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
