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
│   ├── ground_truth/        # Validated "gold standard" CSVs
│   ├── clean_ground_truth/  # Cleaned & anchored output CSVs
│   └── logs/
│       └── difflib/         # Diff reports from anchoring pipeline
├── notebooks/
│   ├── extraction/          # Ground truth extraction notebooks
│   │   ├── edgar_ground_truth_cleanup.ipynb  # Main cleanup pipeline
│   │   ├── llama_3.3_70B_instruct/
│   │   ├── qwen_2.5_32B_instruct/
│   │   └── gemini/
│   │   └── _archived_prompts/
│   └── experimentation/     # Experiments with the ground truth data
├── docs/
│   ├── plan/                # Pipeline plans & design docs
│   ├── Beyond_Haystack_RS_Paper.pdf
│   └── reference/           # Partner's code (read-only reference or other paper's code)
```

### Reference Repositories (Snapshots)

The `docs/reference/` directory contains frozen snapshots of partner code to ensure reproducibility. To avoid nested `.git` issues, these are maintained as static copies.
* **Ananya's Code:** `docs/reference/ananya-code/` (Upstream: https://github.com/4n4ny4/iahd_experiments)
* **Harry's Code:** `docs/reference/harry-code/` (Upstream: https://github.com/harryila/iahdClean/tree/main)

*Note: These directories are excluded in `.geminiignore` to prevent AI context pollution, but can be referenced explicitly.*

## 📊 Data Naming Conventions

### Extracted Data (`data/extracted/`)

Format: `{script_name}_{num_rows}_{MM-DD-YYYY}.csv`
Example: `tournament_full_250_12-26-2025.csv`

### Ground Truth (`data/ground_truth/`)

Format: `v{version}_{num_rows}_{MM-DD-YYYY}.csv`
Example: `v1_250_1-6-2025.csv`

## 🛠️ Development Guidelines

- **Prefer Notebooks:** Write logic in notebooks or standalone scripts. No package abstraction yet.
- **Hardware:** We run on Lambda Labs GPUs (A100-40GB or H100s). Code should be efficient. If we are using Gemini API Key, we wouldn't have to worry about the hardware, but for the opens weighted models we do.  

## Best Practices

- Adhere to SOLID principles.
- Avoid hard-coded values — use config files, environment variables, or centralized constants.
- Remove dead/unused code; keep implementations minimal and focused.
- Prefer small, single-responsibility functions and clear, descriptive names.
- Comment intent ("why"), not implementation ("what"); keep comments concise. 
- Before calling a tool or running a command, explain why you are calling it and what you expect it to do.

### File Modification Rules

- **GEMINI.md & README.md:** May modify without permission.
- **All other files:** Ask for permission before making changes. (If changes are to be made put them in the side panel and wait for approval.)
- **For Data Investigation or Analysis:** Generate the code in the side panel and ask for it to be implemented in the `notebooks/extraction/ai-investigation.ipynb`

### When Unsure

Ask for clarification rather than guessing.
