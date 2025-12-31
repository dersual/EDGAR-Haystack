# EDGAR-Haystack

Recreating "Beyond the Haystack" paper findings using SEC 10-K filings and Llama-3-8B-Instruct on Lambda Cloud GPUs.

## Project Overview

This repository houses scripts and experiments designed to extract specific data points from the EDGAR Corpus dataset (SEC 10-K filings). The primary research goal is to investigate how Large Language Models (LLMs) retrieve information from financial documents, particularly under "shuffled" text conditions (recreating findings from the "Beyond the Haystack" paper).

**Core Question**: Do models perform differently on shuffled vs unshuffled financial text? Which question types show interesting patterns?

## Current Status

**Primary Focus: Data Extraction via Notebooks**

Previous attempts to build a formal Python package were premature and resulted in unnecessary complexity. We have shifted to a notebook-centric workflow to rapidly iterate on extraction logic.

- **Active Work**: `notebooks/Beyond_The_Haystack_Recreation_Using_Edgar_V2.ipynb` is the current working baseline.
- **Goal**: reliably extracting fields (e.g., Incorporation State, Employee Count) from 10-K filings to verify our Ground Truth dataset before proceeding to the shuffling experiments.

## Directory Guide

| Directory | Description |
|-----------|-------------|
| **`notebooks/`** | **Active Development**. Contains the primary notebooks for data extraction and experimentation. |
| **`csvs/`** | Stores data artifacts. `extracted_data/` contains outputs, and `ground_truth/` contains validated datasets used to benchmark performance. |
| **`gt_extract/`** | **Reference Code**. Partner's work used for context. Contains scripts and logic used to generate the initial ground truth data. |
| **`_archive/`** | **Deprecated**. Contains over-engineered code from previous iterations (e.g., the old `edgar_haystack` package). Do not use. |

## Infrastructure & Workflow

We utilize **Lambda Labs** for our computational resources.
- **Hardware**: Experiments are run on Lambda Cloud GPUs (typically A10 or H100 instances).
- **Access**: Work is performed via SSH or by directly uploading Jupyter notebooks to the remote instance.
- **Environment**: The environment uses standard PyTorch/HuggingFace libraries.

### Quick Start on Lambda Cloud

#### 1. Launch GPU Instance
```bash
export LAMBDA_API_KEY="your_key_here"

curl --user "${LAMBDA_API_KEY}:" \
  --request POST "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
  --data '{
    "region_name":"us-west-1",
    "instance_type_name":"gpu_1x_a10",
    "ssh_key_names":["your-ssh-key"],
    "name":"edgar-experiment"
  }'
```

#### 2. Get Instance URL
```bash
# Save instance ID from previous command, then:
curl --user "${LAMBDA_API_KEY}:" \
  --request GET "https://cloud.lambda.ai/api/v1/instances/<INSTANCE_ID>"
```

#### 3. Open Jupyter and Run Notebook
- Open the Jupyter URL from the response
- Upload `notebooks/Beyond_The_Haystack_Recreation_Using_Edgar_V2.ipynb`
- Run cells

## The Experiment Methodology

### What We're Testing
1. Load 10-K financial filings from HuggingFace (`c3po-ai/edgar-corpus`).
2. Create different shuffle conditions (no shuffle → local → global).
3. Ask questions about the documents (Single-instance words, Financial facts, Dates, etc.).
4. Compare accuracy across conditions.

### Key Insight from Paper
The paper found models do **better** on globally shuffled (incoherent) text for certain tasks. This suggests they're not really "reading" but pattern matching. We want to see if this holds for financial documents.

## References
- Paper: [Beyond the Haystack](https://aclanthology.org/2025.nllp-1.5.pdf)
- Dataset: [c3po-ai/edgar-corpus](https://huggingface.co/datasets/c3po-ai/edgar-corpus)
- Answer extraction approach: [iahd repo](https://github.com/harryila/iahd)