# Data Files

This directory contains the generated JSONL files and image assets referenced by those files.

- `input_code.jsonl`: code-only input setting. `GT w/o anno chart` is always `null`.
- `input_code_image.jsonl`: code-plus-image input setting. `GT w/o anno chart` points to the removed-annotation chart image.
- `input_image_only.jsonl`: image-only input setting. `GT w/o anno chart` points to the removed-annotation chart image; `GT w/o anno code` is always `null`.
- `manifest.json`: generated schema and row-count metadata.
- `images/GT_chart/`: annotated target charts.
- `images/GT_w_o_anno_chart/`: charts before annotation.

Regenerate these files from the project root with:

```bash
python github_release/scripts/build_github_jsonl.py
```

