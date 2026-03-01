# Implementation Plan: Retrieval Head Ablation (Step 2)

## Context

Step 1 (Sum Attention identification) is complete in [retrieval_head_identification.ipynb](file:///c:/Users/dersu/Code Stuff/EDGAR-Haystack/notebooks/experimentation/llama_3.3_70B_instruct/summed_attention/retrieval_head_identification.ipynb). It produces:

- Per-task top-K retrieval head rankings
- Cross-task "shared" retrieval heads (heads appearing in ≥7 tasks)
- A `retrieval_heads.json` artifact saved to `data/retrieval_heads/results/`

Step 2 needs to **ablate** these heads and measure the impact on actual retrieval accuracy (NIAH evaluation).

## User Review Required

> [!IMPORTANT] > **Continue from the same notebook or create a new one?**
>
> **Recommendation: Create a new notebook** (`retrieval_head_ablation.ipynb`) in the same directory. Reasons:
>
> 1. **Colab-friendly**: You can upload _just_ the ablation notebook + the `retrieval_heads.json` artifact without rerunning the expensive identification pass.
> 2. **Decoupled**: Step 2 loads the JSON artifact from Step 1, so it can run independently (even with externally-provided head rankings).
> 3. **Cleaner narrative**: The identification notebook stays focused; ablation gets its own story.
> 4. **Shared setup cells**: Auth, config, model loading, and `build_prompt` are identical — we copy them into the new notebook so it's fully self-contained.
>
> The Step 1 notebook remains untouched.

> [!IMPORTANT] > **Ablation scope — what experiments to include?**
>
> Per the [original plan](file:///c:/Users/dersu/Code Stuff/EDGAR-Haystack/docs/plan/summed_attention_head_llama_instruct_plan.md), Step 2 has two experiments:
>
> 1. **Within-task ablation**: For each task, ablate _that task's own_ top-K heads → expect big accuracy drop.
> 2. **Across-task task-specific ablation**: Ablate the top heads from other tasks (i.e., for each task, ablate the top-K heads from all other tasks) rather than their own tasks.
> 3. **Across-task general ablation**: Ablate the _shared/global_ top-K heads (from the intersection) across all tasks → expect a general accuracy drop.
>
> Both are included in this plan. We also add a **random head ablation** control experiment (ablating randomly selected heads instead of the top heads) to prove that the accuracy drop is specific to retrieval heads.

---

## Proposed Changes

### New Notebook

#### [CREATED] [retrieval_head_ablation.ipynb](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/notebooks/experimentation/llama_3.3_70B_instruct/summed_attention/retrieval_head_ablation.ipynb)

The notebook has been created with these sections:

---

**Section 1: Environment Setup (reused from Step 1)**

- `%pip install` cell
- Imports
- Colab / Lambda Labs auth cells
- Configuration cell (same `MODEL_ID`, `TARGET_SEQ_LEN`, `TASKS`, `TASK_MAP`, paths, etc.)
- New config constants:
  ```python
  ABLATION_K_VALUES = [1, 2, 5, 10, 20]     # How many top heads to ablate incrementally
  MAX_DECODE_TOKENS = 20              # Greedy decode length
  HEADS_JSON_PATH = "data/retrieval_heads/results/retrieval_heads.json"
  ABLATION_OUTPUT_DIR = "data/retrieval_heads/ablation_results"
  ```
- Tokenizer & model loading (identical to Step 1)

---

**Section 2: Load Identified Heads**

- Load `retrieval_heads.json` from Step 1 (or an uploaded file on Colab)
- Parse per-task top-K head lists and the shared heads list into [(layer, head)](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_single_depth_haystack_plan.py#16-18) tuples
- Print a summary table: task → number of candidate heads, shared head count

---

**Section 3: Core Ablation Infrastructure**

- **[HeadAblationHooks](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#156-192) context manager** (adapted from Ananya's [run_ablation_shared_heads.py](file:///c:/Users/dersu/Code Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py)):

  - This mechanism intercepts the model's forward pass dynamically during inference.
  - It uses PyTorch's `register_forward_pre_hook` on the attention output projection (`o_proj`) layer.
  - For the specific heads we want to ablate, it mathematically zeros out their activation values in the tensor (setting them to 0). This effectively turns those heads off and stops them from passing information forward, without permanently changing the model weights.
  - Context manager pattern ([**enter**](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#168-187)/[**exit**](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#188-192)) for clean hook registration/removal

- **[greedy_decode](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#121-154) function**:

  - Prefill + incremental decode loop (no attention output needed during decode — saves memory)
  - Returns decoded string

- **[normalize_value_for_match](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_single_depth_haystack_plan.py#127-138) function** (from Ananya's code):
  - Strips commas, converts number words to digits, lowercases
  - Used for "value match" accuracy scoring

---

**Section 4: Prompt Building & Evaluation Helpers**

- Reuse `build_prompt` and [find_needle_span](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_single_depth_haystack_plan.py#38-49) from Step 1 (copy into notebook)

- **`evaluate_sample` function**: Given a row + task + optional ablation heads:

  1. Build prompt with `build_prompt`
  2. Optionally enter [HeadAblationHooks](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#156-192) context
  3. Greedy decode
  4. Score: value_match (needle_value ∈ decoded output)
  5. Return `{"decoded": ..., "value_match": bool}`

- **[evaluate_condition](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#237-327) function**: Loop over `ablation_df` for a given condition (baseline or ablated):
  1. Filter by task (within-task) or all tasks (across-task)
  2. Call `evaluate_sample` for each row
  3. Aggregate: `{task → {attempts, matches, accuracy}}`

---

**Section 5: Experiment 1 — Within-Task Ablation (Task-Specific Heads)**

For each task in the dataset:

1. Run baseline evaluation (no hooks) on `ablation_df` rows for that task
2. Iteratively for each k in `ABLATION_K_VALUES` (e.g., k=1, 2, 5, 10, 20):
   - Hook and ablate the **top-k heads identified specifically for that task**
   - Re-evaluate the model's accuracy on the same rows
3. Collect results into a structured dictionary mapping k-value to accuracy
4. Save raw results JSON

---

**Section 6: Experiment 2 — Across-Task Ablation (General Retrieval Heads)**

This experiment tests if a "general" pool of retrieval heads applies across the board:

1. Baseline accuracy is reused from Experiment 1
2. Iteratively for each k in `ABLATION_K_VALUES` (e.g., k=1, 2, 5, 10, 20):
   - Hook and ablate the **top-k global shared heads** (the heads occurring most frequently across all tasks)
   - Evaluate the model's accuracy across **all** tasks based on this single set of ablated heads
3. Collect results mapping k-value to global accuracy, save JSON

---

**Section 7: Visualizations**

To keep implementation simple, we will stick to basic plots:

1. **Within-task line/bar chart**: Accuracy curve showing Baseline vs decreasing accuracy as k increases through `[1, 2, 5, 10, 20]`.
2. **Across-task aggregate chart**: Overall accuracy drop as the general/shared heads are cumulatively ablated.
3. **Summary table**: A simple DataFrame printed to the notebook showing exact accuracy numbers at each k.
4. All figures saved to `ABLATION_OUTPUT_DIR`.

---

**Section 8: Random Head Ablation Control**

- As a control condition, ablate _randomly selected_ non-retrieval heads across the same increments (k=1, 2, 5, 10, 20).
- If accuracy remains essentially flat/stable when ablating random heads (but drops significantly when cumulatively ablating retrieval heads), it proves that these specific heads are causally responsible for retrieval.

---

## Verification Plan

### On-GPU Validation (Lambda Labs / Colab)

Since ablation requires the actual 70B model loaded on GPU, all testing must happen on the target hardware. The notebook is designed so you can:

1. **Smoke test with `max_rows=2`**: Add a `MAX_ROWS_PER_TASK = 2` override at the top of the config cell, run the entire notebook end-to-end, and verify:

   - [HeadAblationHooks](file:///c:/Users/dersu/Code%20Stuff/EDGAR-Haystack/docs/reference/ananya-code/run_ablation_shared_heads.py#156-192) installs/removes cleanly (no errors)
   - Greedy decode produces non-empty outputs
   - Accuracy dict structure is correct
   - Plots render without errors
   - JSON files are saved

2. **Unit test the hooking mechanism**: A standalone cell that:

   - Installs hooks on 2 random heads
   - Runs a single forward pass
   - Asserts the hooked head's `o_proj` input has zeros in the correct slice
   - Removes hooks and verifies the output is restored

3. **Full run**: Remove the `MAX_ROWS_PER_TASK` cap and run end-to-end on the 20% ablation split

### Manual Verification

- **Review the plots**: After the full run, visually confirm that ablating top retrieval heads causes a meaningful accuracy drop compared to baseline
- **Compare against Ananya's results**: Sanity-check that the magnitude of the drop is in the same ballpark as the reference implementation
