
#!/bin/bash

DOCS_DIR=$1

# Define your list
list=(
    "auth"
    "clone"
    "config"
    "data_frame"
    "df_utils"
    "diff/diff"
    "diff/line_diff"
    "diff/tabular_diff"
    "diff/text_diff"
    "init"
    "oxen_fs"
    "remote_repo"
    "repo"
    "streaming_dataset"
    "user"
    "workspace"
)

# Loop over the list
for i in "${list[@]}"; do
    echo "Generating docs for $i in python-api/$i.mdx"
    # make parent dir if not exists $DOCS_DIR/python-api/$i.mdx
    mkdir -p $DOCS_DIR/python-api/$(dirname $i)

    pydoc-markdown -I python -m oxen.$i --no-render-toc > $DOCS_DIR/python-api/$i.mdx

    # for our docs we want the class methods to be top level in the nav
    gsed -i "s:#### :## :g" $DOCS_DIR/python-api/$i.mdx
done
