import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


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


def find_needle_span(prompt_ids, needle_ids):
    span_len = len(needle_ids)
    needle_set = set(needle_ids)
    if span_len == 0:
        return -1, -1
    for i in range(len(prompt_ids)):
        span = prompt_ids[i : i + span_len]
        overlap = len(set(span) & needle_set) / max(1, len(needle_set))
        if overlap > 0.9:
            return i, i + span_len
    return -1, -1


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


def retrieval_calculate(attentions, retrieval_score, inp_id, prompt_ids, needle_start, needle_end, topk=1):
    for layer_idx in range(len(attentions)):
        head_dim = attentions[layer_idx].shape[1]
        for head_idx in range(head_dim):
            values, idx = attentions[layer_idx][0][head_idx][-1].topk(topk)
            for i in idx:
                if needle_start <= i < needle_end and inp_id == prompt_ids[i].item():
                    retrieval_score[layer_idx][head_idx] += 1 / (needle_end - needle_start)
                    break


def greedy_decode_with_retrieval(model, tokenizer, prompt_ids, needle_ids, max_decode):
    needle_start, needle_end = find_needle_span(prompt_ids[0].tolist(), needle_ids)
    if needle_start < 0:
        return "", None

    retrieval_score = np.zeros(
        (model.config.num_hidden_layers, model.config.num_attention_heads),
        dtype=float,
    )

    with torch.no_grad():
        # Do not request attentions on prefill: with long prompts this is O(seq^2)
        # and can OOM. We only need attentions during incremental decode below.
        outputs = model(input_ids=prompt_ids[:, :-1], use_cache=True, return_dict=True, output_attentions=False)
        past_kv = outputs.past_key_values
        inp = prompt_ids[:, -1:]
        # Explicit position_ids for incremental decode (avoids RoPE shape mismatch with cache)
        current_position = prompt_ids.size(1) - 1
        device = prompt_ids.device

        generated = []
        for _ in range(max_decode):
            position_ids = torch.tensor(
                [[current_position]],
                dtype=torch.long,
                device=device,
            )
            out = model(
                input_ids=inp,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_id = out.logits[:, -1].argmax(dim=-1)
            generated.append(next_id.item())
            retrieval_calculate(out.attentions, retrieval_score, next_id.item(), prompt_ids[0], needle_start, needle_end)

            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
            current_position += 1
            inp = next_id.unsqueeze(1)

    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return decoded, retrieval_score


def head_set_from_scores(avg_scores, threshold):
    heads = set()
    for l in range(avg_scores.shape[0]):
        for h in range(avg_scores.shape[1]):
            if avg_scores[l, h] >= threshold:
                heads.add((l, h))
    return heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haystack_csv", required=True)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_decode", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--target_tokens", type=int, default=7000)
    parser.add_argument("--output_dir", default="experiments/outputs/single_depth")
    parser.add_argument("--max_rows", type=int, default=None, help="Max CSV rows to process (default: all). Use for quick tests.")
    parser.add_argument(
        "--max_rows_per_task",
        type=int,
        default=100,
        help="Max CSV rows to process per task (default: 100).",
    )
    parser.add_argument(
        "--gate",
        choices=["value", "rouge", "hybrid"],
        default="rouge",
        help="Gate for which rows contribute to retrieval head scores: value=value match only, "
        "rouge=ROUGE-1 recall vs needle > threshold, hybrid=value match AND rouge > threshold (default: rouge).",
    )
    parser.add_argument(
        "--rouge_threshold",
        type=float,
        default=0.5,
        help="ROUGE-1 recall threshold for rouge/hybrid gate (default: 0.5, i.e. 50%%).",
    )
    args = parser.parse_args()

    if args.gate in ("rouge", "hybrid") and rouge_scorer is None:
        raise RuntimeError("--gate rouge/hybrid requires rouge-score package. Install with: pip install rouge-score")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.output_dir, f"run_{timestamp}"))
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

    task_scores = {}
    task_success = {}
    task_attempts = {}
    task_value_match = {}
    task_rouge_ok = {}
    task_rows_seen = {}

    rouge_scorer_obj = None
    if args.gate in ("rouge", "hybrid"):
        rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    target_tasks = set(TASK_QUESTIONS.keys())
    capped_target_tasks = set()
    total_rows = 0
    final_rows = []

    with open(args.haystack_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="CSV rows", file=sys.stderr):
            if len(capped_target_tasks) == len(target_tasks):
                break
            total_rows += 1
            if args.max_rows is not None and total_rows > args.max_rows:
                break
            task = row.get("task")
            needle = row.get("needle_sentence", "")
            needle_value = row.get("needle_value", "")
            haystack = row.get("haystack_text", "")
            if not task or not needle or not haystack:
                continue

            task_scores.setdefault(task, [])
            task_success.setdefault(task, 0)
            task_attempts.setdefault(task, 0)
            task_value_match.setdefault(task, 0)
            task_rouge_ok.setdefault(task, 0)
            task_rows_seen.setdefault(task, 0)
            if task_rows_seen[task] >= args.max_rows_per_task:
                continue
            task_rows_seen[task] += 1
            if task in target_tasks and task_rows_seen[task] >= args.max_rows_per_task:
                capped_target_tasks.add(task)

            if normalize_text(needle) in normalize_text(haystack):
                continue

            question = TASK_QUESTIONS.get(task, f"What is the {task}?")
            context = insert_needle_at_depth(tokenizer, haystack, needle, question, 50)
            context_tokens = tokenizer.encode(context)
            target_len = args.target_tokens
            if len(context_tokens) > target_len:
                context = tokenizer.decode(context_tokens[:target_len])
            elif len(context_tokens) < target_len:
                padding_tokens = tokenizer.encode(" " * (target_len - len(context_tokens)))
                context = tokenizer.decode(context_tokens + padding_tokens)

            if normalize_text(needle) not in normalize_text(context):
                continue
            task_attempts[task] += 1
            final_rows.append(
                {
                    "filename": row.get("filename", ""),
                    "task": task,
                    "needle_sentence": needle,
                    "needle_value": needle_value,
                    "haystack_text": haystack,
                    "context_with_needle": context,
                    "needle_in_haystack": normalize_text(needle) in normalize_text(haystack),
                }
            )

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

            needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
            decoded, score = greedy_decode_with_retrieval(model, tokenizer, inputs, needle_ids, args.max_decode)
            if score is None:
                continue

            value_match = bool(
                needle_value
                and normalize_value_for_match(needle_value) in normalize_value_for_match(decoded)
            )
            rouge_ok = False
            if rouge_scorer_obj is not None:
                rouge_result = rouge_scorer_obj.score(needle, decoded)
                rouge_recall = rouge_result["rouge1"].recall
                rouge_ok = rouge_recall >= args.rouge_threshold
                if rouge_ok:
                    task_rouge_ok[task] += 1

            if value_match:
                task_value_match[task] += 1

            if args.gate == "value":
                gate_pass = value_match
            elif args.gate == "rouge":
                gate_pass = rouge_ok
            else:
                gate_pass = value_match and rouge_ok

            if gate_pass:
                task_scores[task].append(score)
                task_success[task] += 1

    os.makedirs(run_dir, exist_ok=True)
    task_heads = {}
    task_head_rankings = {}
    task_avg_scores = {}
    for task in TASK_QUESTIONS:
        scores_list = task_scores.get(task, [])
        if not scores_list:
            task_heads[task] = []
            task_head_rankings[task] = []
            task_avg_scores[task] = None
            continue
        avg_scores = np.mean(np.stack(scores_list, axis=0), axis=0)
        task_avg_scores[task] = avg_scores
        ranked_heads = []
        for l in range(avg_scores.shape[0]):
            for h in range(avg_scores.shape[1]):
                score = float(avg_scores[l, h])
                if score >= args.threshold:
                    ranked_heads.append(
                        {
                            "head": [l, h],
                            "avg_score": score,
                        }
                    )
        ranked_heads.sort(key=lambda x: x["avg_score"], reverse=True)
        task_head_rankings[task] = ranked_heads
        task_heads[task] = sorted([h["head"] for h in ranked_heads])

    with open(os.path.join(run_dir, "task_heads.json"), "w") as f:
        json.dump(task_heads, f, indent=2)

    with open(os.path.join(run_dir, "task_success.json"), "w") as f:
        json.dump(task_success, f, indent=2)

    with open(os.path.join(run_dir, "task_attempts.json"), "w") as f:
        json.dump(task_attempts, f, indent=2)

    with open(os.path.join(run_dir, "task_value_match.json"), "w") as f:
        json.dump(task_value_match, f, indent=2)

    if task_rouge_ok:
        with open(os.path.join(run_dir, "task_rouge_ok.json"), "w") as f:
            json.dump(task_rouge_ok, f, indent=2)

    with open(os.path.join(run_dir, "task_head_rankings.json"), "w") as f:
        json.dump(task_head_rankings, f, indent=2)

    final_csv = os.path.join(run_dir, "final_experiment_rows.csv")
    with open(final_csv, "w", newline="") as f:
        fieldnames = [
            "filename",
            "task",
            "needle_sentence",
            "needle_value",
            "haystack_text",
            "context_with_needle",
            "needle_in_haystack",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "threshold": args.threshold,
                "max_decode": args.max_decode,
                "gate": args.gate,
                "rouge_threshold": args.rouge_threshold if args.gate in ("rouge", "hybrid") else None,
                "total_rows": total_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
