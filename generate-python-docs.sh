
#!/bin/bash

DOCS_DIR=$1

# Define your list
list=("remote_repo" "local_repo")

# Loop over the list
for i in "${list[@]}"; do
    echo "Generating docs for $i in python-api/$i.mdx"
    pydoc-markdown -I python -m oxen.$i --no-render-toc > $DOCS_DIR/python-api/$i.mdx
    # for our docs we want the class methods to be top level in the nav
    gsed -i "s:#### :## :g" $DOCS_DIR/python-api/$i.mdx
done
