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

Navigate into your local [oxen-release](https://github.com/Oxen-AI/oxen-release) project and into the `oxen` directory so that it can generate from the doc strings.

```
# Nagivate into the oxen-release codebase where the python doc strings are
cd ~/Code/docs/oxen-release/oxen

# Generate the markdown for the classes
~/Code/docs/generate-python-docs.sh ~/Code/docs/
```
