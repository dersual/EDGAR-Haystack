import argparse
import csv
import json
import os
import re
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TASK_QUESTIONS = {
    "registrant_name": "What is the registrant name?",
    "headquarters_city": "What is the headquarters city?",
    "headquarters_state": "What is the headquarters state?",
    "incorporation_state": "What is the incorporation state?",
    "incorporation_year": "What is the incorporation year?",
    "employees_count_total": "What is the total employee count?",
    "employees_count_full_time": "What is the full-time employee count?",
    "ceo_lastname": "What is the CEO's last name?",
    "holder_record_amount": "What is the holder record amount?",
}


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip().lower()


# Number normalization for value-match so "1,000" / "1000" and "eight" / "8" match.
# Built from number words and compounds that appear in data/haystack_plan_100_per_task.csv
# (needle_sentence) and that the model may output for numeric tasks.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
}
# Hyphenated compounds (e.g. twenty-one -> 21); applied first so longer forms win.
_NUMBER_WORDS_COMPOUNDS = {
    "twenty-one": "21", "twenty-two": "22", "twenty-three": "23", "twenty-four": "24",
    "twenty-five": "25", "twenty-six": "26", "twenty-seven": "27", "twenty-eight": "28", "twenty-nine": "29",
    "thirty-one": "31", "thirty-two": "32", "thirty-three": "33", "thirty-four": "34",
    "thirty-five": "35", "thirty-six": "36", "thirty-seven": "37", "thirty-eight": "38", "thirty-nine": "39",
    "forty-one": "41", "forty-two": "42", "forty-three": "43", "forty-four": "44",
    "forty-five": "45", "forty-six": "46", "forty-seven": "47", "forty-eight": "48", "forty-nine": "49",
    "fifty-one": "51", "fifty-two": "52", "fifty-three": "53", "fifty-four": "54",
    "fifty-five": "55", "fifty-six": "56", "fifty-seven": "57", "fifty-eight": "58", "fifty-nine": "59",
    "sixty-one": "61", "sixty-two": "62", "sixty-three": "63", "sixty-four": "64",
    "sixty-five": "65", "sixty-six": "66", "sixty-seven": "67", "sixty-eight": "68", "sixty-nine": "69",
    "seventy-one": "71", "seventy-two": "72", "seventy-three": "73", "seventy-four": "74",
    "seventy-five": "75", "seventy-six": "76", "seventy-seven": "77", "seventy-eight": "78", "seventy-nine": "79",
    "eighty-one": "81", "eighty-two": "82", "eighty-three": "83", "eighty-four": "84",
    "eighty-five": "85", "eighty-six": "86", "eighty-seven": "87", "eighty-eight": "88", "eighty-nine": "89",
    "ninety-one": "91", "ninety-two": "92", "ninety-three": "93", "ninety-four": "94",
    "ninety-five": "95", "ninety-six": "96", "ninety-seven": "97", "ninety-eight": "98", "ninety-nine": "99",
}


def normalize_value_for_match(text):
    """Normalize text for value-match: lowercase, collapse whitespace, strip commas, number words -> digits."""
    if text is None:
        return ""
    s = normalize_text(text)
    s = re.sub(r",", "", s)
    for phrase, digit in _NUMBER_WORDS_COMPOUNDS.items():
        s = re.sub(r"\b" + re.escape(phrase) + r"\b", digit, s)
    for word, digit in _NUMBER_WORDS.items():
        s = re.sub(r"\b" + re.escape(word) + r"\b", digit, s)
    return s


def insert_needle_at_depth(tokenizer, context, needle, question, depth_percent, buffer_tokens=None):
    tokens_context = tokenizer.encode(context)
    tokens_needle = tokenizer.encode(needle)

    if buffer_tokens is None:
        messages = [
            {
                "role": "user",
                "content": f"<document></document>\nBased on the content of the document, Question: {question}\nAnswer:",
            }
        ]
        base_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        base_len = base_ids.shape[-1] if hasattr(base_ids, "shape") else len(base_ids)
        buffer_tokens = base_len + len(tokens_needle)

    target_len = max(0, len(tokens_context) - buffer_tokens)
    if len(tokens_context) + len(tokens_needle) > target_len:
        tokens_context = tokens_context[: max(0, target_len - len(tokens_needle))]

    if depth_percent >= 100:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_new_context = tokens_context[:insertion_point]
        period_tokens = tokenizer.encode(".")
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    return tokenizer.decode(tokens_new_context)


def greedy_decode(model, tokenizer, prompt_ids, max_decode):
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
        past_kv = outputs.past_key_values
        inp = prompt_ids[:, -1:]
        current_position = prompt_ids.size(1) - 1
        device = prompt_ids.device

        generated = []
        for _ in range(max_decode):
            position_ids = torch.tensor([[current_position]], dtype=torch.long, device=device)
            out = model(
                input_ids=inp,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_id = out.logits[:, -1].argmax(dim=-1)
            generated.append(next_id.item())
            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
            current_position += 1
            inp = next_id.unsqueeze(1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


class HeadAblationHooks:
    """Zero selected attention head channels before o_proj."""

    def __init__(self, model, heads):
        self.model = model
        self.heads = heads
        self.handles = []
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.by_layer = {}
        for l, h in heads:
            self.by_layer.setdefault(l, set()).add(h)

    def __enter__(self):
        for layer_idx, layer_heads in self.by_layer.items():
            if layer_idx >= len(self.model.model.layers):
                continue
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj

            def pre_hook(_module, inputs, heads=sorted(layer_heads)):
                x = inputs[0]
                if x.ndim != 3:
                    return inputs
                x = x.clone()
                for head_idx in heads:
                    start = head_idx * self.head_dim
                    end = start + self.head_dim
                    x[..., start:end] = 0
                return (x,)

            self.handles.append(o_proj.register_forward_pre_hook(pre_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def build_shared_head_list(task_rankings, max_heads=10):
    """Build top shared heads ranked by frequency across tasks. Used for zero-head fallback."""
    head_counts = {}
    for task, heads in task_rankings.items():
        for h in heads:
            head_counts[h] = head_counts.get(h, 0) + 1
    ranked = sorted(head_counts.items(), key=lambda x: (-x[1], x[0]))
    return [h for h, _ in ranked[:max_heads]]


def build_global_head_ranking(task_rankings, max_heads=10):
    """Build global top heads (by frequency across tasks) for across-task ablation."""
    return build_shared_head_list(task_rankings, max_heads)


def load_task_head_rankings(rankings_json_path):
    with open(rankings_json_path, "r") as f:
        payload = json.load(f)

    task_rankings = {}
    if isinstance(payload, dict):
        for task, entries in payload.items():
            parsed = []
            for entry in entries or []:
                head = entry.get("head") if isinstance(entry, dict) else entry
                if isinstance(head, list) and len(head) == 2:
                    parsed.append((int(head[0]), int(head[1])))
            task_rankings[task] = parsed
    elif isinstance(payload, list):
        # Backward compatibility: global candidate list gets reused for each task.
        global_heads = []
        for entry in payload:
            head = entry.get("head") if isinstance(entry, dict) else entry
            if isinstance(head, list) and len(head) == 2:
                global_heads.append((int(head[0]), int(head[1])))
        for task in TASK_QUESTIONS:
            task_rankings[task] = list(global_heads)
    else:
        raise ValueError(f"Unsupported ranking JSON format: {rankings_json_path}")

    return task_rankings


def evaluate_condition(
    model,
    tokenizer,
    args,
    condition_name,
    ablation_heads=None,
    task_filter=None,
):
    task_attempts = {}
    task_success = {}
    task_value_match = {}
    rows_seen_per_task = {}
    rows_scanned = 0

    hook_ctx = HeadAblationHooks(model, ablation_heads) if ablation_heads else nullcontext()
    with hook_ctx:
        with open(args.haystack_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"{condition_name} rows"):
                rows_scanned += 1
                if args.max_rows is not None and rows_scanned > args.max_rows:
                    break

                task = row.get("task")
                needle = row.get("needle_sentence", "")
                needle_value = row.get("needle_value", "")
                haystack = row.get("haystack_text", "")
                if not task or not needle or not haystack:
                    continue
                if task_filter is not None and task != task_filter:
                    continue

                rows_seen_per_task.setdefault(task, 0)
                if rows_seen_per_task[task] >= args.max_rows_per_task:
                    continue
                rows_seen_per_task[task] += 1

                task_attempts.setdefault(task, 0)
                task_success.setdefault(task, 0)
                task_value_match.setdefault(task, 0)

                if normalize_text(needle) in normalize_text(haystack):
                    continue

                question = TASK_QUESTIONS.get(task, f"What is the {task}?")
                context = insert_needle_at_depth(tokenizer, haystack, needle, question, 50)
                context_tokens = tokenizer.encode(context)
                if len(context_tokens) > args.target_tokens:
                    context = tokenizer.decode(context_tokens[: args.target_tokens])
                elif len(context_tokens) < args.target_tokens:
                    padding_tokens = tokenizer.encode(" " * (args.target_tokens - len(context_tokens)))
                    context = tokenizer.decode(context_tokens + padding_tokens)

                if normalize_text(needle) not in normalize_text(context):
                    continue

                task_attempts[task] += 1
                messages = [
                    {
                        "role": "user",
                        "content": f"<document>{context}</document>\nBased on the content of the document, Question: {question}\nAnswer:",
                    }
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
                if hasattr(inputs, "input_ids"):
                    inputs = inputs["input_ids"].to(model.device)
                else:
                    inputs = inputs.to(model.device)

                decoded = greedy_decode(model, tokenizer, inputs, args.max_decode)
                value_match = bool(
                    needle_value
                    and normalize_value_for_match(needle_value) in normalize_value_for_match(decoded)
                )
                if value_match:
                    task_value_match[task] += 1
                    task_success[task] += 1

    return {
        "rows_scanned": rows_scanned,
        "task_attempts": task_attempts,
        "task_success": task_success,
        "task_value_match": task_value_match,
        "ablation_heads": [list(x) for x in (ablation_heads or [])],
    }


def summarize_per_task_results(per_task_results, k_values):
    summary = {
        "k_values": k_values,
        "per_task": {},
        "overall_by_k": {},
    }

    overall_attempts = {k: 0 for k in k_values}
    overall_baseline_value = {k: 0 for k in k_values}
    overall_ablated_value = {k: 0 for k in k_values}

    for task, task_results in per_task_results.items():
        baseline = task_results["baseline"]
        b_attempts = sum(baseline["task_attempts"].values())
        b_value = sum(baseline["task_success"].values())
        b_rate = b_value / max(1, b_attempts)

        task_summary = {
            "available_heads": task_results["available_heads"],
            "baseline": {
                "attempts": b_attempts,
                "value_match": b_value,
                "value_rate": b_rate,
            },
            "conditions": {},
        }

        for k in k_values:
            cond_name = f"k_{k}"
            cond = task_results["conditions"][cond_name]
            c_attempts = sum(cond["task_attempts"].values())
            c_value = sum(cond["task_success"].values())
            c_rate = c_value / max(1, c_attempts)
            k_eff = len(cond["ablation_heads"])

            task_summary["conditions"][cond_name] = {
                "k_requested": k,
                "k_effective": k_eff,
                "attempts": c_attempts,
                "value_match": c_value,
                "value_rate": c_rate,
                "delta_value_rate_vs_baseline": c_rate - b_rate,
            }

            overall_attempts[k] += c_attempts
            overall_baseline_value[k] += b_value
            overall_ablated_value[k] += c_value

        summary["per_task"][task] = task_summary

    for k in k_values:
        att = max(1, overall_attempts[k])
        base_rate = overall_baseline_value[k] / att
        ablated_rate = overall_ablated_value[k] / att
        summary["overall_by_k"][f"k_{k}"] = {
            "attempts": overall_attempts[k],
            "baseline_value_rate": base_rate,
            "ablated_value_rate": ablated_rate,
            "delta_value_rate_vs_baseline": ablated_rate - base_rate,
        }

    return summary


def plot_per_task_value_ablation(summary, out_dir):
    tasks = sorted(summary["per_task"].keys())
    if not tasks:
        return
    k_values = summary["k_values"]

    delta = np.zeros((len(k_values), len(tasks)))
    for i, k in enumerate(k_values):
        key = f"k_{k}"
        for j, task in enumerate(tasks):
            delta[i, j] = summary["per_task"][task]["conditions"][key][
                "delta_value_rate_vs_baseline"
            ]

    fig, ax = plt.subplots(figsize=(max(11, len(tasks) * 1.1), max(4.5, len(k_values) * 1.2)))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f"k={k}" for k in k_values])
    ax.set_title("Within-Task: Delta Accuracy vs Baseline (pp)")
    for i in range(len(k_values)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Delta Accuracy (pp)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_within_task_delta_heatmap.png"), dpi=150)
    plt.close(fig)

    n_tasks = len(tasks)
    n_bars = 1 + len(k_values)  # Baseline + k conditions
    bar_width = 0.18  # same width for every bar
    group_width = n_bars * bar_width + (n_bars - 1) * 0.08  # bars + small gaps
    x = np.arange(n_tasks) * (group_width + 0.4)  # even spacing between task groups
    fig, ax = plt.subplots(figsize=(max(12, n_tasks * 1.4), 6))
    baseline_acc = [
        100 * summary["per_task"][t]["baseline"]["value_rate"] for t in tasks
    ]
    offsets = [-1.5 * bar_width - 0.12, -0.5 * bar_width - 0.04, 0.5 * bar_width + 0.04, 1.5 * bar_width + 0.12]
    ax.bar(x + offsets[0], baseline_acc, bar_width, label="Baseline", color="#2E86AB", edgecolor="white", linewidth=0.8)
    k_colors = ["#28A745", "#FD7E14", "#6F42C1"]
    for i, k in enumerate(k_values):
        acc = [
            100 * summary["per_task"][t]["conditions"][f"k_{k}"]["value_rate"]
            for t in tasks
        ]
        ax.bar(x + offsets[i + 1], acc, bar_width, label=f"k={k}", color=k_colors[i % 3], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Within-Task: Ablating Each Task's Top Heads", fontsize=12, fontweight="medium")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_within_task_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(k_values))
    baseline_rates = [
        100 * summary["overall_by_k"][f"k_{k}"]["baseline_value_rate"] for k in k_values
    ]
    ablated_rates = [
        100 * summary["overall_by_k"][f"k_{k}"]["ablated_value_rate"] for k in k_values
    ]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_rates, width, label="Baseline", color="#4C72B0")
    ax.bar(x + width / 2, ablated_rates, width, label="Ablated (within-task)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Within-Task: Aggregate Accuracy Across All Tasks")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_within_task_aggregate.png"), dpi=150)
    plt.close(fig)


def plot_across_task_accuracy(across_task_results, tasks, k_values, out_dir):
    x = np.arange(len(k_values))
    total_baseline_val = sum(across_task_results["baseline"][t]["value_match"] for t in tasks)
    total_baseline_att = sum(across_task_results["baseline"][t]["attempts"] for t in tasks)
    baseline_agg = total_baseline_val / max(1, total_baseline_att)
    ablated_agg = []
    for k in k_values:
        total_val = sum(across_task_results["by_k"][k]["task_value_match"].get(t, 0) for t in tasks)
        total_att = sum(across_task_results["by_k"][k]["task_attempts"].get(t, 0) for t in tasks)
        ablated_agg.append(total_val / max(1, total_att))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        [100 * baseline_agg] * len(k_values),
        width,
        label="Baseline",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        [100 * r for r in ablated_agg],
        width,
        label="Ablated (across-task)",
        color="#C44E52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Across-Task: Ablating Global Top 1, 5, 10 Heads")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_across_task_accuracy.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haystack_csv", required=True)
    parser.add_argument(
        "--candidate_heads_json",
        required=True,
        help="Path to task_head_rankings.json (preferred) or legacy ablation_candidates.json",
    )
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_decode", type=int, default=20)
    parser.add_argument("--target_tokens", type=int, default=7000)
    parser.add_argument("--max_rows_per_task", type=int, default=100)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--ablation_k",
        default="1,5,10",
        help="Comma-separated ablation sizes (default: 1,5,10)",
    )
    parser.add_argument("--output_dir", default="experiments/outputs/ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

    k_values = [int(x.strip()) for x in args.ablation_k.split(",") if x.strip()]
    if not k_values:
        raise ValueError("No valid ablation k values were provided.")

    task_head_rankings = load_task_head_rankings(args.candidate_heads_json)
    all_task_names = sorted(task_head_rankings.keys())
    tasks = sorted(t for t in all_task_names if len(task_head_rankings.get(t, [])) > 0)
    skipped = [t for t in all_task_names if t not in tasks]
    if skipped:
        print(f"Skipping {len(skipped)} task(s) with no retrieval heads: {skipped}")
    if not tasks:
        print("No tasks with retrieval heads; nothing to ablate.")
        return

    per_task_results = {}
    for task in tasks:
        ranked_heads = task_head_rankings.get(task, [])
        per_task_results[task] = {
            "available_heads": len(ranked_heads),
            "baseline": evaluate_condition(
                model=model,
                tokenizer=tokenizer,
                args=args,
                condition_name=f"{task} baseline",
                ablation_heads=None,
                task_filter=task,
            ),
            "conditions": {},
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for k in k_values:
            k_eff = min(k, len(ranked_heads))
            ablation_heads = ranked_heads[:k_eff]
            cond_name = f"k_{k}"
            per_task_results[task]["conditions"][cond_name] = evaluate_condition(
                model=model,
                tokenizer=tokenizer,
                args=args,
                condition_name=f"{task} {cond_name}",
                ablation_heads=ablation_heads,
                task_filter=task,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_per_task_results(per_task_results, k_values)

    global_heads = build_global_head_ranking(task_head_rankings, max_heads=max(max(k_values), 10))
    across_task_results = {"baseline": {}, "by_k": {}}
    for task in tasks:
        baseline = per_task_results[task]["baseline"]
        across_task_results["baseline"][task] = {
            "attempts": sum(baseline["task_attempts"].values()),
            "value_match": sum(baseline["task_success"].values()),
        }
    for k in k_values:
        k_eff = min(k, len(global_heads))
        ablation_heads = global_heads[:k_eff]
        cond = evaluate_condition(
            model=model,
            tokenizer=tokenizer,
            args=args,
            condition_name=f"across-task k={k}",
            ablation_heads=ablation_heads,
            task_filter=None,
        )
        across_task_results["by_k"][k] = {
            "task_attempts": cond["task_attempts"],
            "task_value_match": cond["task_value_match"],
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_across_task_accuracy(across_task_results, tasks, k_values, run_dir)

    with open(os.path.join(run_dir, "condition_results.json"), "w") as f:
        json.dump(per_task_results, f, indent=2)
    with open(os.path.join(run_dir, "across_task_results.json"), "w") as f:
        json.dump(across_task_results, f, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "per_task_ablation_summary.json"), "w") as f:
        json.dump(summary["per_task"], f, indent=2)
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "max_decode": args.max_decode,
                "target_tokens": args.target_tokens,
                "max_rows_per_task": args.max_rows_per_task,
                "ablation_k": k_values,
                "ranking_source": args.candidate_heads_json,
                "tasks": tasks,
                "skipped_tasks": skipped,
            },
            f,
            indent=2,
        )

    plot_per_task_value_ablation(summary, run_dir)
    print(f"Ablation run saved to: {run_dir}")


if __name__ == "__main__":
    main()
