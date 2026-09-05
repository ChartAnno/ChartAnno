<p align="center">
  <img src="assets/apple-touch-icon.png" width="140">
</p>

<h1 align="center">ChartAnno: Evaluating MLLMs for Chart Annotation Generation</h1>

<p align="center">
  <a href="https://chartanno.github.io/"><img src="https://img.shields.io/badge/Homepage-chartanno.github.io-teal.svg" alt="Homepage"></a>
  <a href="https://arxiv.org/abs/2608.03464"><img src="https://img.shields.io/badge/arXiv-2608.03464-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/chartanno/ChartAnno"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-ffd21f.svg" alt="HuggingFace Dataset"></a>
  <br>
  <img src="https://img.shields.io/badge/Code%20License-Apache--2.0-blue.svg" alt="Code License">
  <img src="https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg" alt="Data License">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
</p>

## 🎉 What's New

- __[2026.09]__ 📣 ChartAnno dataset and codebase are released.
- __[2026.09]__ ➕ We extend the benchmark with 120 D3 and SVG charts as a first step toward the generalizability of our evaluation framework.

## 🎏 Introduction

Annotations are essential to communicative visualization, helping explain data, emphasize key findings, and guide attention. While multimodal large language models (MLLMs) offer new opportunities for automatic chart annotation authoring, their capabilities in this task remain underexplored. To address this gap, we introduce **ChartAnno**, a comprehensive benchmark for evaluating MLLMs on **chart annotation generation**.

- **1,200 real-world charts** with paired annotated and unannotated executable code.
- **3,600 annotation instructions** spanning three levels of specificity (Intent, Operation, Implementation).
- A **multidimensional evaluation framework** combining rule-based and LLM-judged metrics to assess execution, structural compliance, semantic consistency, and design effectiveness.
- **10 representative MLLMs** evaluated.

<p align="center">
  <img src="assets/academic_motivation.png" width="95%">
</p>
<p align="center"><sub>Figure 1. Motivating example of chart annotation generation. Given an existing chart and an annotation instruction, an MLLM generates executable code to add annotations that communicate the intended message.</sub></p>

Evaluating 10 representative MLLMs reveals that **proprietary models lead overall**, though open-source models narrow the gap. While **higher instruction specificity improves annotation quality**, inferring abstract communicative intent remains difficult across all models. Providing chart images yields marginal benefit when code is available. Further analyses uncover common failure modes, the impact of visual complexity, and the validity of LLM-as-a-judge protocols. Experiments with **D3 and SVG** demonstrate the **generalizability** of ChartAnno beyond its primary Python setting.

<p align="center">
  <img src="assets/main-chart.png" width="100%">
</p>
<p align="center"><sub>Figure 2. Task design of ChartAnno. The benchmark considers two primary input settings, Code and Code + Image, and three instruction levels: Intent, Operation, and Implementation. The MLLM produces generated code, which is rendered into the generated chart.</sub></p>

## 📄 Table of Contents

<details open>
<summary>Click to expand the table of contents</summary>

- 🎉 What's New
- 🎏 Introduction
- 🚀 Quick Start
  - Setup Environment
  - Download Data
  - Evaluate Models
- 📚 Data
- 🧪 Evaluation
- 💬 Citation
- 📌 License

</details>

## 🚀 Quick Start

### Setup Environment

```bash
cd ChartAnno
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_API_KEY"   # or any OpenAI-compatible provider, see below
```

### Download Data

The dataset is hosted on HuggingFace and is not bundled with this repository. It is gated — click **Request access** on the [repo page](https://huggingface.co/datasets/chartanno/ChartAnno) first, then download with a read token ([settings/tokens](https://huggingface.co/settings/tokens)):

```bash
pip install -U huggingface_hub
hf login --token $HF_TOKEN
hf download chartanno/ChartAnno chartanno.tar.gz --repo-type dataset --local-dir .
tar -xzvf chartanno.tar.gz    # extracts to ./data
```

### Evaluate Models

Run generation for both input settings (`MODE=llm` for `Input: code`, `MODE=vlm` for `Input: code+Image`):

```bash
MODE=both MODEL=gpt-5.4 bash tasks/run_tasks.sh
```

Or run the full pipeline (generation → rendering → rule-based evaluation → LLM-judged evaluation → score merging):

```bash
./run_pipeline.sh \
  --steps all \
  --model gpt-5.4 \
  --source-model gpt54 \
  --base-url https://api.openai.com/v1
```

To run only specific instruction levels, add `--levels intent` (or `--levels intent,operation`); the rest of the pipeline processes only what was generated.

<details>
<summary>Useful environment variables</summary>

| Variable | Default | Description |
| --- | --- | --- |
| `MODE` | `both` | `llm`, `vlm`, or `both` for the shell wrapper |
| `MODEL` | `gpt-5.4` | Model name passed to the API |
| `BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `API_KEY_ENV` | `OPENAI_API_KEY` | Environment variable containing the API key |
| `MAX_DATA_ROWS` | `0` | Row limit; `0` means all rows |
| `LEVELS` | `intent,operation,implementation` | Level filter |
| `CATEGORIES` | empty | Optional comma-separated category filter |

</details>

For a guided walkthrough, see [QUICKSTART.md](QUICKSTART.md).

## 📚 Data

ChartAnno contains **1,200 chart instances**. Each instance includes:

- A ground-truth (GT) pair with annotations (code + rendered image).
- A GT pair without annotations (code + rendered image).

Each chart is paired with three annotation instructions across the **Intent, Operation, and Implementation** levels, resulting in **3,600 tasks**. The primary benchmark tests two input settings (`Input: code` and `Input: code+Image`) — **7,200 model input instances**; a supplementary `Input: Image` setting adds **3,600 image-only instances**, and a 120-chart D3.js / SVG extension adds **720** more, spanning **three representations** (Python, D3.js, SVG).

After extracting `chartanno.tar.gz`, the `data/` folder has the following layout (only `data/README.md` and `data/manifest.json` are checked into this repository):

```text
data/
├── input_code.jsonl              # Input: code task, 3,600 rows
├── input_code_image.jsonl        # Input: code+Image task, 3,600 rows
├── input_image_only.jsonl        # Input: Image task, 3,600 rows
├── manifest.json                 # row-count summary and schema description
├── README.md                     # data folder documentation
└── images/
    ├── GT_chart/                 # 1,200 annotated ground-truth charts (jpg)
    │   ├── Area/  Bar/  ...      # one subdirectory per chart type (17 types)
    └── GT_w_o_anno_chart/        # 1,200 unannotated charts, same layout
```

Each JSONL row has the following fields:

| Field | Description |
| --- | --- |
| `id` | Stable row id: `<sample_id>_<level>_code` or `<sample_id>_<level>_code_image`. |
| `category` | Chart type, one of 17 types (e.g. `Area`, `Bar`, `Line`). |
| `sample_id` | Source chart id, `<Category>_<n>` (e.g. `Area_1`). |
| `level` | Instruction level: `intent`, `operation`, or `implementation`. |
| `input_type` | `Input: code` or `Input: code+Image`. |
| `instruction` | The full model input prompt: task instruction with the unannotated code embedded. |
| `GT w/o anno code` | Ground-truth chart code before annotation. |
| `GT w/o anno chart` | Path to the unannotated chart image; always `null` in `input_code.jsonl`. |
| `GT code` | Ground-truth annotated chart code. |
| `GT chart` | Path to the annotated ground-truth chart image. |

**Overall scale**

| Statistic | Value |
| --- | --- |
| GT pair and GT w/o anno pair | 1,200 |
| Instruction instances | 3,600 |
| Annotation elements | 25,772 |

The full dataset (all 3,600 rows per config, chart images embedded as compressed thumbnails) is browsable in the [HuggingFace Dataset Viewer](https://huggingface.co/datasets/chartanno/ChartAnno/viewer/1_python_code_image/train) (`1/2/3_python_*` configs = main benchmark; `4_d3_code_only` / `5_svg_code_only` = extension).

### D3.js / SVG extension (code-only)

Besides the main Python benchmark, a **D3.js and SVG extension** is available on HuggingFace: **120 charts** (14 types) × 3 instruction levels per backend, **360 rows each** (`4_d3_code_only` / `5_svg_code_only` configs; full package in `chartanno_d3_svg.tar.gz`). Rows follow the same schema as `2_python_code_only`, with `.js` / `.svg` code and `_d3` / `_svg` id suffixes.

## 🧪 Evaluation

The evaluation pipeline runs in five stages: **generation** (the model writes annotated chart code) → **rendering** (generated code is executed to produce chart images) → **rule-based evaluation** → **LLM-judged evaluation** → **score merging**.

Execution and rule-based metrics:

```text
execution_success_rate, chart_fidelity, annotation_matching, color_matching
```

LLM-judged metrics:

```text
semantic_faithfulness, semantic_clarity,
visual_clarity, annotation_organization_quality, attention_guidance
```

The final aggregate scores are computed as:

```text
structural_compliance = chart_fidelity                                     # intent level
structural_compliance = mean(chart_fidelity, annotation_matching, color_matching)
semantic_consistency  = mean(semantic_faithfulness, semantic_clarity)
design_effectiveness  = mean(visual_clarity, annotation_organization_quality, attention_guidance)
```

Rule-based scores and `execution_success_rate` are in `[0, 1]`; LLM-judged scores are in `[0, 5]`. Missing/invalid outputs are counted as `0`. See [QUICKSTART.md](QUICKSTART.md) for the full command reference and output layout.

To examine the generalizability of our evaluation framework, we apply the same framework to D3 and SVG, adapting *Graphical Element Extraction* and *Differential Annotation Extraction and Classification* in the rule-based evaluation to each representation. The adapted evaluators are provided in [evaluation/rule_based_d3](evaluation/rule_based_d3) and [evaluation/rule_based_svg](evaluation/rule_based_svg).

## 💬 Citation

If you find this repository useful, please consider giving a star and citing our paper:

```bibtex
@article{chen2026chartanno,
      title={ChartAnno: Evaluating MLLMs for Chart Annotation Generation},
      author={Zhenghan Chen and Zekai Shao and Lidan Tan and Xin Lin and Xingchen Zeng and Yi Shan and Ziyue Lin and Xiaoliang Fu and Xinyuan Liu and Yuetong Guo and Fen Wang and Bongshin Lee and Siming Chen},
      year={2026},
      journal={arXiv preprint arXiv:2608.03464},
}
```

## 📌 License

The code is released under [Apache-2.0](LICENSE). The dataset is released separately under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
