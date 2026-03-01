# Pipeline Design: Retrieval Head Identification & Ablation in Llama-3.3-70B-Instruct

## Overview

This document outlines a clean, modular Jupyter Notebook pipeline designed to identify, validate, and ablate "retrieval heads" in the `meta-llama/Llama-3.3-70B-Instruct` model. By combining the "Sum Attention" identification method with "Needle in a Haystack" (NIAH) causal validation, this pipeline transitions from messy research scripts to a structured, reproducible mechanistic interpretability workflow.

---

## Environment Setup & Initialization

**Objective:** Establish a secure, reproducible, and memory-efficient environment for analyzing a 70B parameter model.

1. **Authentication & Configuration:**
   - **Colab vs. Lambda Labs:** Support both environments. Use `python-dotenv` to load `.env` files for local/Lambda Labs setups. If in Google Colab, fallback to `google.colab.userdata.get('HF_TOKEN')`.
   - **Hugging Face Login:** Programmatically log in using `huggingface_hub.login(token=HF_TOKEN)` to ensure access to gated models like Llama-3.3.
   - Define global configuration variables (e.g., `MODEL_ID`, `TARGET_SEQ_LEN`, `BATCH_SIZE`).
2. **Model & Tokenizer Loading:**
   - Initialize the tokenizer.
   - **Precision:** Prefer loading the model in full/half precision (bfloat16/float16) if VRAM permits (e.g., multi-GPU Lambda Labs). Provide 4-bit/8-bit quantization (via `bitsandbytes`) only as an optional fallback if memory constrained.
   - **Crucial:** Ensure the model is loaded with `output_attentions=True` to natively expose attention matrices during the forward pass.
3. **Modularization:**
   - Keep all implementation (helper classes, data processing, metric calculation) contained within well-organized, distinct cells in the Jupyter Notebook for now to simplify the workflow.

---

## Step 1: Identification Logic (Sum Attention)

**Objective:** Systematically isolate which attention heads are responsible for retrieving information from the context using Harry's "Sum Attention" methodology.

1. **Data Preparation & Splitting:**
   - **Dataset Split:** Split the dataset 80/20. Use the 80% split strictly for identifying the retrieval heads, and reserve the 20% split for validation/ablation tasks.
   - **Insertion Strategy (Ananya vs. Harry):**
     - _Ananya's Approach:_ Dynamically insert the needle at a fixed depth (e.g., 50%) and pad/truncate to an exact token length (e.g., 7000 tokens).
       - _Pros:_ Highly controlled, uniform attention matrix dimensions (great for batching), isolates depth as a variable.
       - _Cons:_ Artificial context boundaries, might cut off text abruptly.
     - _Harry's Approach (Phase 1):_ Pre-cache realistic document sections (e.g., EDGAR Section 1) with natural needle placements.
       - _Pros:_ More realistic to actual RAG tasks, faster runtime since it's pre-computed.
       - _Cons:_ Variable lengths make batching attention matrices trickier.
     - _Decision:_ We will lean towards Ananya's approach for the identification phase to ensure strict control over token lengths and depth, making attention extraction mathematically uniform.
2. **Needle Span Tracking:**
   - Tokenize the prompt and programmatically locate the exact start and end token indices of the inserted needle (`needle_start` to `needle_end`).
3. **Forward Pass & Attention Extraction:**
   - Run the model on the prepared prompts.
   - Extract the raw attention weights from all layers and heads.
4. **Sum Attention Scoring:**
   - For every head in every layer, isolate the attention weights originating from the **last token** (the query token predicting the first token of the answer).
   - Calculate the score: `score(layer, head) = Σ attention[last_token, needle_start:needle_end]`.
   - This quantifies how intensely each head "looks" at the exact location of the needle.
5. **Aggregation & Ranking (Task-Specific vs. General):**
   - _Context from prior work:_ Harry identifies heads _per task_ first, then analyzes overlap in later phases. Ananya also identifies per task, but then explicitly ablates "shared heads" across all tasks.
   - **Our Approach:**
     1. Calculate and rank the top-K retrieval heads **per task** (e.g., `employee_count`, `hq_state`).
     2. Compute the **intersection** of these top heads across all tasks to identify the "General Retrieval Heads".
     3. **Save Artifacts:** Export these identified heads to a JSON/CSV file (similar to Harry's `tokens_X.json` outputs). This decouples Step 1 from Step 2, allowing us to load externally identified heads if needed.

---

## Step 2: Validation & Ablation Logic

**Objective:** Causally prove that the identified heads are responsible for retrieval (Ananya's NIAH approach) and explore targeted ablation for performance/efficiency gains.

1. **Load Identified Heads:**
   - Read the JSON/CSV file generated in Step 1 (or an external file provided by other researchers). This ensures Step 2 can run independently of Step 1.
2. **Baseline NIAH Evaluation:**
   - Run the model on the NIAH dataset _without_ any modifications.
   - Record baseline accuracy (e.g., exact match, ROUGE, or LLM-as-a-judge) across different context depths.
3. **Targeted Head Ablation:**
   - Implement a hooking mechanism (e.g., PyTorch `register_forward_pre_hook` on the `o_proj` layer of specific attention heads).
   - When the hook triggers, dynamically zero-out the output of the top-K identified retrieval heads.
4. **Causal Validation (Proving Necessity):**
   - Re-run the NIAH evaluation with the top-K retrieval heads ablated.
   - **Expected Result:** A catastrophic drop in retrieval accuracy, causally proving these specific heads are necessary for fetching long-context information.
5. **Efficiency & Performance Boost (Inverse Ablation / Pruning):**
   - Conversely, ablate the _lowest-scoring_ (non-retrieval) heads during a retrieval-heavy task.
   - Measure if accuracy remains stable while compute/memory overhead decreases, demonstrating a pathway to model compression or dynamic routing for efficiency.

---

## Expected Deliverables

1. **Clean Jupyter Notebook:** A well-documented, step-by-step notebook (`retrieval_head_identification.ipynb`) that executes the pipeline without cluttered script logic.
2. **Ranked Head Artifacts:** A JSON or CSV file listing the top retrieval heads (Layer, Head Index, Average Sum Attention Score).
3. **Visualizations:**
   - **Attention Heatmaps:** Visualizing the last token's attention across the context, highlighting the spike at the needle span.
   - **Ablation Impact Charts:** Bar charts or line graphs comparing Baseline Accuracy vs. Ablated Accuracy across different context depths.
   - **Head Distribution:** A scatter plot or grid showing where retrieval heads are localized (e.g., middle vs. late layers).
