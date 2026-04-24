# Oxen Docs

Source for the Mintlify-hosted docs site at [docs.oxen.ai](https://docs.oxen.ai).

## Authoring example terminal sessions

When a page shows a command and its terminal output, put the command and the output in
**separate fenced code blocks**. The copy button on Mintlify code blocks copies the entire
block contents verbatim, so mixing commands and output in one block means readers copy
output (and any `$` prompt) along with the command.

- The command block is what the reader should be able to click-copy and paste into a shell.
  Tag it with the appropriate language (`` ```bash ``, `` ```python ``, etc.) and include
  **only** the command(s) — no leading `$` prompt, no interleaved output.
- The output block holds the terminal output. Use an untagged fence (`` ``` ``) so it renders
  as plain preformatted text without syntax highlighting. Nothing in it should be copy-pasted
  back into a shell.

### Do

````markdown
```bash
oxen df data.tsv
```

```
shape: (4_774, 2)
+-----------+---------------------------------+
| category  | text                            |
+-----------+---------------------------------+
| ham       | Go until jurong point...        |
+-----------+---------------------------------+
```
````

### Don't

````markdown
```bash
$ oxen df data.tsv

shape: (4_774, 2)
+-----------+---------------------------------+
| category  | text                            |
+-----------+---------------------------------+
| ham       | Go until jurong point...        |
+-----------+---------------------------------+
```
````

The "don't" version makes the copy button copy `$ oxen df data.tsv` plus the whole table.

Equally wrong — and much more common in practice — is leaving the command and output in
*separate* blocks but tagging the output block as `` ```bash ``. The copy button still sits
on it, and readers who click it paste a table or log into their shell. If a block contains
no runnable command, it must be untagged.

### Multiple command/output pairs

If a section demonstrates several commands in sequence, make each one its own
command/output pair rather than a single giant block. Shell comments (`# ...`) that explain
a command belong inside the command block, on the line before the command.

### Config files and structured output

When the "output" is a file listing, a config file, or other structured content, tag that
second block with the file's format (`toml`, `json`, `yaml`, ...) for syntax highlighting;
still keep it separate from the command that produced it.
