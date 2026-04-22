# Oxen.ai Documentation 🐂 🌾

Install the mint CLI tool: https://www.mintlify.com/docs/installation

```bash
npm i -g mint
```

## Running Locally

```bash
mint dev --port 3333
```

## Finetune API Documentation Generation

initialize uv environment by running `uv init` and then run `uv sync` to install dependencies.
then `uv run generate-finetune-docs.py` to generate the finetune api documentation.

This will fetch the new models from https://hub.oxen.ai/api/evaluations/models and generate the documentation from them. The documentation will be generated in the `fine-tuning-api/reference` directory.

## Per-Model Inference API Reference Pages

```
uv run generate-model-docs.py
```

Fetches the same `/api/evaluations/models` list and writes one `.mdx` page per inference-capable model under `inference-api/reference/models/`, plus an index page at `inference-api/reference/model-references.mdx` that links to each one (grouped by developer). Only the index page is listed in `mint.json`; the per-model pages are reachable by direct URL. This avoids Mintlify's eager render of a 130-leaf flat nav that made the Inference API tab slow to load. Pass `--input path/to/models.json` to run offline against a saved payload.

## Python Doc Generation

To generate/update the python documentation for the `/python-api` directory there is a `./generate-python-docs.sh` script.

Depends on [pydoc-markdown](https://pypi.org/project/pydoc-markdown/) project and `gsed`.

```
brew install gsed
pip install pydoc-markdown
```

Navigate into your local [Oxen](https://github.com/Oxen-AI/Oxen) project and into the `oxen-python` directory so that it can generate from the doc strings.

```
# Nagivate into the Oxen codebase where the python doc strings are
cd ~/Code/Oxen/Oxen/oxen-python

# Generate the markdown for the classes
~/Code/Oxen/docs/generate-python-docs.sh ~/Code/Oxen/docs/
```

## OpenAPI Spec Generation

We need a file that has the OpenAPI spec in JSON format that is served from a public server.

```
postman2openapi ~/Downloads/OxenServer\ API.postman_collection.json -f json | jq '.servers = [{"url": "https://hub.oxen.ai"}, {"url": "http://localhost:3001"}]' > ~/Code/Oxen/OxenHub/priv/static/api/oxen-server-openapi-spec.json
```
