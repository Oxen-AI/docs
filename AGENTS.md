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
- **Put an `Output:` line between them.** Mintlify automatically groups two adjacent fenced
  code blocks into a single tabbed block (as if they were wrapped in `<CodeGroup>`), even if
  there's a blank line between them. A plain-prose `Output:` paragraph between the command
  and output blocks is enough non-code content to break the grouping, and it also labels the
  second block for the reader.

### Do

````markdown
```bash
oxen df data.tsv
```

Output:

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

## Generated pages

The pages under `python-api/` are generated from the `oxen-python` docstrings in the [Oxen](https://github.com/Oxen-AI/Oxen) repo by `generate-python-docs.sh`, so an edit made to one of them here is lost the next time anyone runs it. Correct the docstring in that repo instead, then regenerate. The README has the invocation. It needs `pydoc-markdown` and GNU `sed` (as `gsed`) on `PATH`.

- **Regenerate whenever an `oxen-python` docstring changes**, including when a class gains a method or property. Nothing else publishes a docstring, and these pages have sat more than a year behind the package that ships.
- **The script has no per-page flag.** It rewrites every page in its list, so the change carries each page whose docstrings moved rather than only the one you came for.

`fine-tuning-api/reference/` and the model reference are generated as well, by `generate-finetune-docs.py` and `generate-model-docs.py`, and the model pages refresh on a weekly workflow. The same rule applies: change the source, not the page.

## Public

This repository is **public**. Do not mention Oxen's private/internal repositories — by name or description — in code comments, doc-comments, error messages, commit messages, PR titles or descriptions, or any other code or documentation committed here. Keep references to private repos out of public artifacts entirely; if internal context is genuinely needed, point to the relevant Linear issue rather than inlining private-repo details.

## GitHub Actions

Pin actions in `.github/workflows/` by trust tier, following the [org-wide policy](https://github.com/Oxen-AI/Oxen/blob/main/docs/github_actions_pinning.md) — version tag for first-party/high-trust orgs, full commit SHA with a `# vX.Y.Z` comment for third-party.
