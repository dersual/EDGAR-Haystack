# EDGAR-Haystack

A dual-phase research repository focused on dataset creation from SEC 10-K filings and the subsequent mechanistic interpretability of Retrieval Heads in Large Language Models.

## Project Overview

This repository is split into two primary research phases:

1.  **Dataset Creation (Extraction):** Constructing a 100% accurate, "gold standard" ground truth dataset from the EDGAR Corpus of SEC 10-K filings.
2.  **Mechanistic Interpretability:** Investigating how models like Llama-3-8B and Llama-3.3-70B retrieve information from long, structured financial contexts. By identifying "retrieval heads" and performing targeted ablation, we aim to causally prove how attention mechanisms operate on naturalistic haystacks.

This work extends upon the foundational findings of the *Beyond the Haystack* paper, integrating newer methodologies for retrieval head identification.

## Current Status

**Phase 4: Mechanistic Interpretability & Attention Ablation**

We are actively identifying, ranking, and ablating attention heads using the **Summed Attention** methodology. 

- **Active Work:** `notebooks/experimentation/llama_3_8B_instruct/summed_attention/`
- **Models:** Llama-3-8B-Instruct (local/Colab) and Llama-3.3-70B-Instruct (Multi-GPU).
- **Environment:** Google Colab GPUs and Lambda Labs Cloud GPUs (A100-40GB / H100s).

## Pipeline Cycles

### 1. Data Extraction Cycle
1.  **Ingestion:** Load 10-K filings from HuggingFace (`c3po-ai/edgar-corpus`).
2.  **Extraction:** Prompt LLMs to extract specific data points (Incorporation State, Employee Count, etc.).
3.  **Refinement:** Clean, normalize, and validate extracted data against raw text to form a "clean ground truth" dataset.

### 2. Experimentation Cycle (Mechanistic Interpretability)
1.  **Identification (`01_extract_attention.ipynb`):** Perform forward passes on clean ground truth data to extract raw attention weights and score heads via Summed Attention.
2.  **Ranking (`02_rank_heads.ipynb`):** Aggregate scores to rank heads per-task and identify globally shared "powerhouse" heads.
3.  **Ablation (`03_run_ablation.ipynb`):** Causal validation by zeroing out targeted heads during generation on held-out data.
4.  **Analysis (`04_analysis_visualizations.ipynb`):** Generate heatmaps and metrics to interpret the causal role of identified heads.

## Directory Structure

```
EDGAR-Haystack/
├── data/
│   ├── clean_ground_truth/  # Cleaned, anchored dataset for experiments
│   └── retrieval_heads/     # Outputs for the 4-step MI pipeline
│       ├── 01_extractions/  # Raw attention tensors (.npy)
│       ├── 02_rankings/     # Ranked JSON lists of heads
│       ├── 03_ablations/    # Ablation metrics (JSON)
│       └── 04_analysis_plots/ # Plots and Heatmaps (PNG)
├── notebooks/
│   ├── extraction/          # Ground truth extraction & cleanup
│   └── experimentation/     # Mechanistic Interpretability (NIAH, Ablation)
│       └── llama_3_8B_instruct/
│           └── summed_attention/
├── docs/
│   ├── plan/                # Architecture and pipeline plans
│   └── reference/           # Research papers and partner code snapshots
```

## Quick Start (Hardware Requirements)

Specialized hardware is required due to the VRAM footprint of attention extraction:
- **Llama-3-8B:** A100 (40GB) or L4 (24GB) via Colab or Lambda.
- **Llama-3.3-70B:** Multi-GPU (2x A100-40GB+) required for full precision extraction.

## References

- [Beyond the Haystack](https://aclanthology.org/2025.nllp-1.5.pdf)
- [Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking](https://arxiv.org/pdf/2404.15574)
- [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/pdf/2506.09944)
- Dataset: [c3po-ai/edgar-corpus](https://huggingface.co/datasets/c3po-ai/edgar-corpus)
