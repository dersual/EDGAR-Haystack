# EDGAR-Haystack

This repository explores a reproduction of the *Beyond the Haystack* (NLLP 2025) study, adapted to long-form **financial documents** using the **EDGAR-CORPUS (10-K filings)** dataset and **small open-weight instruction models**.

The goal is to evaluate how well a model such as **Meta-Llama-3-8B-Instruct** can retrieve and reason over sparse information embedded deep within long SEC filings, following the experimental spirit of the original paper.

---

## Context & Goal

The original *Beyond the Haystack* paper studies the ability of large language models to retrieve a specific “needle” of information from long contexts.

This project reproduces that setup with:
- Real-world financial text (EDGAR 10-K filings)
- A smaller, open-weight instruction-tuned model
- Commodity cloud GPU hardware

The emphasis is on **reproducibility, long-context behavior, and failure analysis**, rather than pushing absolute performance.

---

## Reference Paper

**Beyond the Haystack: Evaluating the Retrieval Capabilities of Large Language Models**  
NLLP 2025  
https://aclanthology.org/2025.nllp-1.5.pdf

---

## Dataset

- **EDGAR-CORPUS**
- Source: `c3po-ai/edgar-corpus` (Hugging Face)
- Content: SEC 10-K financial filings

The dataset is used to construct long contexts into which target facts or statements are embedded at varying depths.

---

## Model

- **Meta-Llama-3-8B-Instruct**
- Hugging Face: `meta-llama/Meta-Llama-3-8B-Instruct`

No fine-tuning is assumed initially; the model is evaluated in a zero-shot or prompt-based setting.

---

## Compute Environment

- **Lambda Cloud GPU**
- Hardware: A10 or A100
- Experiments are designed to be runnable on a single GPU.

---

## Task Description (High-Level)

Each evaluation example consists of:
1. A long EDGAR 10-K filing
2. A target piece of information embedded within the document
3. A query requiring retrieval or identification of that information

Model outputs are analyzed for correctness, partial retrieval, or failure to locate the embedded content.

---

## Repository Structure (Tentative)

```text
edgar-haystack/
├── data/           # dataset processing and constructed examples
├── scripts/        # experiment and evaluation scripts
├── notebooks/      # exploratory analysis and visualization
├── experiments/    # configs and outputs
└── README.md
