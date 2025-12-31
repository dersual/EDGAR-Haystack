# Context for AI Assistants (GEMINI.md)

## 🧠 Project Context & "The Story So Far"

**Goal:** Recreate and extend the findings of the paper *"Beyond the Haystack"* using financial data (SEC 10-K filings).
**Core Hypothesis:** The original paper suggests LLMs often ignore context and rely on positional heuristics. We want to see if this holds true for financial audits (e.g., does shuffling the text of a 10-K make the model fail to find the "State of Incorporation"?).

### 📜 History of the Repo

1.  **Phase 1 (The Over-Engineering):**
    *   We initially tried to build a robust Python package (`edgar_haystack`) with separate modules for `needles`, `haystacks`, `models`, and `evaluators`.
    *   **Result:** FAILED. It was too complex too soon. We spent more time fixing imports than running experiments. This code is now in `_archive`. **Do not use it.**

2.  **Phase 2 (The Partner's Code):**
    *   A research partner provided working scripts (`gt_extract/`) that successfully extract data using LLMs (Llama-3).
    *   **Result:** USEFUL. We use this as a reference implementation. If you are stuck on *how* to extract a specific field, look here first.

3.  **Phase 3 (Current - Notebook Mode):**
    *   We are currently working in `notebooks/` to validate our data.
    *   **Problem:** Before we can test "shuffling", we need a perfect "Ground Truth". We need to know *exactly* what the correct answer is for a set of documents.
    *   **Current Task:** We are refining extraction prompts to build a 100% accurate `ground_truth.csv`.

## 🛠️ Operational Logic

### Data Flow
1.  **Input:** EDGAR Corpus (HuggingFace).
2.  **Processing:** Llama-3 (via Lambda Labs) extracts specific fields:
    *   Incorporation State
    *   Fiscal Year End
    *   Employee Count
3.  **Validation:** Extracted data is compared against `csvs/ground_truth/`.

### Development Guidelines for AI
*   **Prefer Notebooks:** For now, write logic in valid standalone Python scripts or Notebook cells. Do not try to abstract into a package yet.
*   **Check `gt_extract`:** Before inventing a new regex or prompt, check if the partner code in `gt_extract` already solved it.
*   **Ignore `_archive`:** Pretend the `_archive` folder does not exist unless you are explicitly looking for a "what not to do" example.
*   **Hardware Awareness:** We run on remote GPUs. Code should be efficient but doesn't need to be edge-optimized. We have VRAM.

## 📍 Key Locations
*   **`notebooks/Beyond_The_Haystack_Recreation_Using_Edgar_V2.ipynb`**: The main active experiment.
*   **`csvs/ground_truth/`**: The "Gold Standard" we are trying to achieve.
*   **`gt_extract/extract_ground_truth.py`**: The reference logic for extraction.
