# Oxen.ai Documentation 🐂 🌾

## Running Locally

```bash
mintlify dev --port 3333
```

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