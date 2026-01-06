# EDGAR-Haystack

Recreating "Beyond the Haystack" paper findings using SEC 10-K filings on Lambda Cloud GPUs.

## Project Overview

This repository extracts data points from SEC 10-K filings (EDGAR Corpus) to investigate how LLMs retrieve information from financial documents under "shuffled" text conditions.

**Core Question:** Do models perform differently on shuffled vs unshuffled financial text?

## Current Status

**Phase 3: Ground Truth Extraction**

We are building a 100% accurate ground truth dataset before running shuffling experiments.

- **Active Work:** `notebooks/extraction/` — per-model extraction notebooks
- **Models:** Qwen 2.5 32B (1x A100-40GB), Llama 3.3 70B (2x A100 required)
- **Frameworks:** vLLM and Transformers variants available

## Directory Structure

```
EDGAR-Haystack/
├── data/
│   ├── extracted/           # LLM extraction outputs
│   └── ground_truth/        # Validated "gold standard" CSVs
├── notebooks/
│   ├── extraction/          # Ground truth extraction
│   │   ├── llama_3.3_70B_instruct/
│   │   ├── qwen_2.5_32B_instruct/
│   │   └── _archived_prompts/
│   └── experimentation/     # NIAH shuffling experiments
├── docs/
│   ├── Beyond_Haystack_RS_Paper.pdf
│   └── reference/           # Partner's reference code
```

## Data Naming Conventions

| Type             | Format                               | Example                              |
| ---------------- | ------------------------------------ | ------------------------------------ |
| **Extracted**    | `{script}_{rows}_{MM-DD-YYYY}.csv`   | `tournament_full_250_12-26-2025.csv` |
| **Ground Truth** | `v{version}_{rows}_{MM-DD-YYYY}.csv` | `v1_250_1-6-2025.csv`                |

## Quick Start (Lambda Cloud)

### 1. Launch Instance

- **1x A100-40GB** → Qwen 2.5 32B (4-bit quantization)
- **2x A100-40GB** → Llama 3.3 70B

### 2. Setup (varies by Lambda image and GPU)

#### If using vLLM

```bash
# If using vLLM (may already be installed depending on image):
pip install vllm datasets pandas tqdm thefuzz python-Levenshtein
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

#### If using transformers

1. Launch IDE on browser
2. Upload notebook
3. Run as appropriately

### 3. Run Extraction

1. Open appropriate notebook from `notebooks/extraction/`
2. For 1x A100-40GB → Use `qwen_2.5_32B_instruct/`
3. For 2x A100 → Use `llama_3.3_70B_instruct/`

## The Experiment

### Methodology

1. Load 10-K filings from HuggingFace (`c3po-ai/edgar-corpus`)
2. Create shuffle conditions (none → local → global)
3. Extract fields (Incorporation State, Employee Count, etc.)
4. Compare accuracy across conditions

### Key Insight from Paper

Models perform **better** on globally shuffled (incoherent) text for certain tasks — suggesting pattern matching over "reading".

## References

- Paper: [Beyond the Haystack](https://aclanthology.org/2025.nllp-1.5.pdf)
- Dataset: [c3po-ai/edgar-corpus](https://huggingface.co/datasets/c3po-ai/edgar-corpus)
