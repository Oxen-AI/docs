'''
Generate Mintlify documentation from fine-tune schemas in the models API response.
'''

import json
import sys
import requests
from typing import Dict, List, Any
from collections import defaultdict


def get_property_description(prop_name: str, prop_schema: Dict[str, Any]) -> str:
    """Generate a description for a property."""
    # Use title first, then description, or fall back to prop_name
    title = prop_schema.get('title', '')
    desc = prop_schema.get('description', '')
    prop_type = prop_schema.get('type', 'unknown')

    parts = []
    # Prefer title over description for the main label
    if title:
        parts.append(title)
    elif desc:
        parts.append(desc)
    
    # Add type info
    if prop_type == 'array':
        items_type = prop_schema.get('items', {}).get('type', 'object')
        parts.append(f"(array of {items_type})")
    elif prop_type == 'integer':
        min_val = prop_schema.get('minimum')
        max_val = prop_schema.get('maximum')
        default = prop_schema.get('default')
        if default is not None:
            parts.append(f"(default: {default})")
        if min_val is not None:
            parts.append(f"(min: {min_val})")
        if max_val is not None:
            parts.append(f"(max: {max_val})")
    elif prop_type == 'number':
        default = prop_schema.get('default')
        if default is not None:
            parts.append(f"(default: {default})")
    elif 'enum' in prop_schema:
        parts.append(f"(options: {', '.join(prop_schema['enum'])})")
    
    return ' '.join(parts) if parts else prop_name


def generate_schema_table(schema: Dict[str, Any]) -> str:
    """Generate a markdown table for a schema's properties."""
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    if not properties:
        return ""
    
    lines = [
        "| Field | Type | Required | Description |",
        "|-------|------|----------|-------------|"
    ]
    
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get('type', 'unknown')
        is_required = "Yes" if prop_name in required else "No"
        description = get_property_description(prop_name, prop_schema)
        
        # Handle special render types
        render_type = prop_schema.get('renderType', '')
        if render_type == 'dfColumn':
            description += " (DataFrame column name)"
        elif render_type == 'multiDFColumn':
            description += " (Multiple DataFrame column names)"
        
        lines.append(f"| `{prop_name}` | {prop_type} | {is_required} | {description} |")
    
    return '\n'.join(lines)


def generate_example_request(schema_name: str, schema: Dict[str, Any]) -> str:
    """Generate an example request body based on the schema."""
    properties = schema.get('properties', {})
    training_params = {}

    for prop_name, prop_schema in properties.items():
        default = prop_schema.get('default')
        if default is not None:
            training_params[prop_name] = default
        else:
            prop_type = prop_schema.get('type', 'string')
            if prop_type == 'string':
                training_params[prop_name] = f"<{prop_name}>"
            elif prop_type == 'integer':
                training_params[prop_name] = prop_schema.get('minimum', 1)
            elif prop_type == 'number':
                training_params[prop_name] = 0.0001
            elif prop_type == 'boolean':
                training_params[prop_name] = True
            elif prop_type == 'array':
                training_params[prop_name] = []

    example = {
        "resource": "main/your-dataset.parquet",
        "base_model": "<model-canonical-name>",
        "script_type": schema_name,
        "training_params": training_params
    }

    return json.dumps(example, indent=2)


def generate_finetune_doc(schema_info: Dict[str, Any], models: List[Dict[str, Any]]) -> str:
    """Generate Mintlify documentation for a fine-tune schema."""
    schema_name = schema_info['name']
    schema_id = schema_info['id']
    schema = schema_info['schema']
    description = schema_info.get('description', '')

    # Find models that use this schema and get their display names and canonical names
    model_list = [(m['display_name'], m['canonical_name']) for m in models if any(
        s['name'] == schema_name for s in m.get('fine_tune_schemas', [])
    )]

    doc = f"""---
title: "Fine-Tune: {schema_name.replace('_', ' ').title()}"
description: "{description}"
---

## Overview

This schema is used for fine-tuning models with **{schema_name.replace('_', ' ')}** capabilities.

### Schema Type

When creating a fine-tune with this schema, use:

```json
{{
  "resource": "main/your-dataset.parquet",
  "base_model": "<model-canonical-name>",
  "script_type": "{schema_name}",
  "training_params": {{
    ...
  }}
}}
```

**Key Parameters:**
- `script_type`: `{schema_name}` (the fine-tune type)
- `base_model`: One of the supported model canonical names below

### Supported Models

{chr(10).join(f'- {name} (`{canonical_name}`)' for name, canonical_name in model_list[:10])}
{f'- ...and {len(model_list) - 10} more' if len(model_list) > 10 else ''}

## Request Schema

### Required Fields

{generate_schema_table(schema)}

## Example Request

<CodeGroup>

```json Request Body
{generate_example_request(schema_name, schema)}
```

```python Python
import requests

url = "https://hub.oxen.ai/api/repos/{{namespace}}/{{repo_name}}/fine_tunes"
headers = {{
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}}

data = {generate_example_request(schema_name, schema).replace('{', '{{').replace('}', '}}')}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

```bash cURL
curl -X POST https://hub.oxen.ai/api/repos/{{namespace}}/{{repo_name}}/fine_tunes \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{generate_example_request(schema_name, schema).replace(chr(10), ' ')}'
```

</CodeGroup>

## Field Details

"""
    
    # Add detailed field descriptions
    properties = schema.get('properties', {})
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get('type', 'unknown')
        title = prop_schema.get('title', '')
        description = prop_schema.get('description', '')
        default = prop_schema.get('default')

        doc += f"\n### `{prop_name}`\n\n"

        # Show title as the main heading label if available
        if title and title != prop_name:
            doc += f"**{title}**\n\n"

        doc += f"**Type:** `{prop_type}`\n\n"

        # Show description if it exists and is different from title
        if description and description != title:
            doc += f"{description}\n\n"
        
        if default is not None:
            doc += f"**Default:** `{json.dumps(default)}`\n\n"
        
        if 'enum' in prop_schema:
            doc += f"**Options:** {', '.join(f'`{v}`' for v in prop_schema['enum'])}\n\n"
        
        if 'minimum' in prop_schema:
            doc += f"**Minimum:** `{prop_schema['minimum']}`\n\n"
        
        if 'maximum' in prop_schema:
            doc += f"**Maximum:** `{prop_schema['maximum']}`\n\n"
    
    return doc


def main():
    # Fetch models from API or read from file
    if len(sys.argv) > 1 and sys.argv[1] != '-':
        # Read from file if provided
        print(f"Reading models from {sys.argv[1]}...")
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
    elif len(sys.argv) > 1 and sys.argv[1] == '-':
        # Read from stdin if '-' is provided
        print("Reading models from stdin...")
        data = json.load(sys.stdin)
    else:
        # Fetch from API by default
        api_url = "https://hub.oxen.ai/api/evaluations/models"
        print(f"Fetching models from {api_url}...")
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Successfully fetched models from API")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models from API: {e}")
            sys.exit(1)

    models = data.get('models', [])
    
    # Extract all unique fine-tune schemas
    schemas_by_name = {}
    for model in models:
        if not model.get('is_fine_tuneable'):
            continue
        
        for schema_info in model.get('fine_tune_schemas', []):
            schema_name = schema_info['name']
            if schema_name not in schemas_by_name:
                schemas_by_name[schema_name] = {
                    'info': schema_info,
                    'models': []
                }
            schemas_by_name[schema_name]['models'].append(model)
    
    print(f"Found {len(schemas_by_name)} unique fine-tune schemas")
    print(f"Found {len([m for m in models if m.get('is_fine_tuneable')])} fine-tuneable models")
    print()
    
    # Generate docs for each schema
    for schema_name, schema_data in schemas_by_name.items():
        filename = f"fine-tuning-api/reference/{schema_name}.mdx"
        print(f"Generating {filename}...")
        
        doc_content = generate_finetune_doc(
            schema_data['info'],
            schema_data['models']
        )
        
        with open(filename, 'w') as f:
            f.write(doc_content)
        
        print(f"  ✓ {len(schema_data['models'])} models use this schema")
    
    print(f"\n✓ Generated {len(schemas_by_name)} documentation files")


if __name__ == '__main__':
    main()
