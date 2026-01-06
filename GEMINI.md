# Context for AI Assistants (GEMINI.md)

## 🧠 Project Context & "The Story So Far"

**Goal:** Recreate and extend the findings of the paper _"Beyond the Haystack"_ using financial data (SEC 10-K filings).

**Core Hypothesis:** The original paper suggests LLMs often ignore context and rely on positional heuristics. We want to see if this holds true for financial audits (e.g., does shuffling the text of a 10-K make the model fail to find the "State of Incorporation"?).

### 📜 Project History

1. **Phase 1 (Over-Engineering):** Tried to build a Python package too early. Failed. Code archived in `docs/reference/`.

2. **Phase 2 (Partner's Code):** Research partner provided working extraction scripts. Now in `docs/reference/gt_extract/` for reference only.

3. **Phase 3 (Current - Notebook Mode):** Working in `notebooks/` to build 100% accurate ground truth before shuffling experiments.

## 📍 Key Locations

```
EDGAR-Haystack/
├── data/
│   ├── extracted/           # LLM extraction outputs
│   └── ground_truth/        # Validated "gold standard" CSVs
├── notebooks/
│   ├── extraction/          # Ground truth extraction notebooks
│   │   ├── llama_3.3_70B_instruct/
│   │   ├── qwen_2.5_32B_instruct/
│   │   └── _archived_prompts/
│   └── experimentation/     # NIAH shuffling experiments
├── docs/
│   ├── Beyond_Haystack_RS_Paper.pdf
│   └── reference/           # Partner's code (read-only reference)
```

## 📊 Data Naming Conventions

### Extracted Data (`data/extracted/`)

Format: `{script_name}_{num_rows}_{MM-DD-YYYY}.csv`
Example: `tournament_full_250_12-26-2025.csv`

### Ground Truth (`data/ground_truth/`)

Format: `v{version}_{num_rows}_{MM-DD-YYYY}.csv`
Example: `v1_250_1-6-2025.csv`

## 🛠️ Development Guidelines

- **Prefer Notebooks:** Write logic in notebooks or standalone scripts. No package abstraction yet.
- **Reference Code:** Check `docs/reference/gt_extract/` before inventing new prompts or regex.
- **Hardware:** We run on Lambda Labs GPUs (A100-40GB). Code should be efficient.

### File Modification Rules

- **GEMINI.md & README.md:** May modify without permission.
- **All other files:** Ask for permission before making changes.
- **Questions about code:** Show proposed changes, wait for approval.

### When Unsure

Ask for clarification rather than guessing.
