#!/usr/bin/env python3
"""
Main evaluation script for EDGAR-Haystack experiment.

This script reproduces the "Beyond the Haystack" methodology using:
1. EDGAR 10-K filings as long documents
2. Financial facts as domain-specific needles
3. Text shuffling to test coherence vs recall
4. Llama-3-8B-Instruct for evaluation

Usage:
    python scripts/run_evaluation.py --num_documents 50 --model_name meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Import our modules
from edgar_haystack import (
    EDGARLoader,
    NeedleExtractor,
    TextShuffler,
    PromptGenerator,
    ResponseScorer,
    LlamaModel,
)


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'edgar_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Run EDGAR-Haystack evaluation")
    parser.add_argument(
        "--num_documents", type=int, default=50, help="Number of documents to evaluate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c3po-ai/edgar-corpus",
        help="EDGAR dataset to use",
    )
    parser.add_argument(
        "--year", type=str, default="2020", help="Year of filings to analyze"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("EDGAR-HAYSTACK EVALUATION")
    logger.info("Reproducing 'Beyond the Haystack' with Financial Documents")
    logger.info("=" * 70)
    logger.info(f"Documents: {args.num_documents}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Methodology: Finding naturally occurring single-instance words")

    # Step 1: Load EDGAR data
    logger.info("\n" + "=" * 50)
    logger.info("STEP 1: LOADING EDGAR DATA")
    logger.info("=" * 50)

    data_loader = EDGARLoader(dataset_name=args.dataset, year=args.year)
    documents = data_loader.get_documents_for_evaluation(num_docs=args.num_documents)

    logger.info(f"Loaded {len(documents)} suitable documents")
    avg_length = sum(doc["word_count"] for doc in documents) / len(documents)
    logger.info(f"Average document length: {avg_length:.0f} words")

    # Step 2: Extract needles (Following Paper's Methodology)
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2: EXTRACTING NATURALLY OCCURRING NEEDLES")
    logger.info("=" * 50)

    needle_extractor = NeedleExtractor(random_seed=42)
    evaluation_cases = []

    for i, doc in enumerate(tqdm(documents, desc="Finding single-occurrence words")):
        # Use paper's methodology: find naturally occurring single-instance words
        needle_case = needle_extractor.create_evaluation_case(doc["text"])

        if needle_case:
            evaluation_cases.append(
                {
                    "document_id": i,
                    "original_text": doc["text"],
                    "needle": needle_case["needle"],
                    "needle_position": needle_case["needle_position"],
                    "words": needle_case["words"],
                    "text_for_shuffling": needle_case["text_for_shuffling"],
                    "word_count": doc["word_count"],
                    "statistics": needle_case["statistics"],
                }
            )

    logger.info(
        f"Successfully extracted needles from {len(evaluation_cases)} documents"
    )

    if not evaluation_cases:
        logger.error(
            "No valid needles found! Check your documents and needle extraction."
        )
        logger.error(
            "Try using longer documents or adjusting needle extraction parameters."
        )
        return

    # Show sample needles
    logger.info("Sample needles found:")
    for i, case in enumerate(evaluation_cases[:3]):
        logger.info(
            f"  {i+1}. '{case['needle']}' (position: {case['needle_position']})"
        )

    logger.info(f"Needle statistics:")
    needle_lengths = [len(case["needle"]) for case in evaluation_cases]
    logger.info(
        f"  Length range: {min(needle_lengths)}-{max(needle_lengths)} characters"
    )
    logger.info(
        f"  Average candidates per doc: {sum(c['statistics']['candidates_found'] for c in evaluation_cases) / len(evaluation_cases):.1f}"
    )

    if not evaluation_cases:
        logger.error(
            "No valid needles found! Check your documents and needle extraction."
        )
        return

    # Step 3: Initialize evaluation components
    logger.info("\n" + "=" * 50)
    logger.info("STEP 3: INITIALIZING EVALUATION")
    logger.info("=" * 50)

    shuffler = TextShuffler(random_seed=42)
    prompt_generator = PromptGenerator()
    scorer = ResponseScorer()

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = LlamaModel(model_name=args.model_name)
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")

    # Step 4: Run evaluation
    logger.info("\n" + "=" * 50)
    logger.info("STEP 4: RUNNING EVALUATION")
    logger.info("=" * 50)

    results = []

    for case in tqdm(evaluation_cases, desc="Evaluating"):
        document_id = case["document_id"]
        needle = case["needle"]
        target_answer = needle
        text_for_shuffling = case["text_for_shuffling"]

        # Create shuffle conditions for this document
        shuffle_conditions = shuffler.create_shuffle_conditions(text_for_shuffling)

        # Test each shuffle condition
        case_results = {
            "document_id": document_id,
            "needle": needle,
            "target_answer": target_answer,
            "word_count": case["word_count"],
        }

        for condition_name, shuffled_text in shuffle_conditions.items():
            # Check context length
            if not model.check_context_length(shuffled_text):
                logger.warning(
                    f"Document {document_id} too long for {condition_name}, truncating"
                )
                shuffled_text = model.truncate_text_to_context(shuffled_text)

            # Generate prompt (using paper's generic methodology)
            messages = prompt_generator.create_chat_messages(shuffled_text)

            # Query model
            try:
                response = model.generate_response(messages, max_new_tokens=30)

                # Score response
                match_result = scorer.check_answer_match(response, target_answer)

                # Store results for this condition
                case_results[f"{condition_name}_response"] = response
                case_results[f"{condition_name}_correct"] = match_result["is_correct"]
                case_results[f"{condition_name}_match_info"] = match_result

            except Exception as e:
                logger.error(
                    f"Error evaluating document {document_id}, condition {condition_name}: {e}"
                )
                case_results[f"{condition_name}_response"] = "ERROR"
                case_results[f"{condition_name}_correct"] = False
                case_results[f"{condition_name}_match_info"] = None

        results.append(case_results)

    # Step 5: Calculate and display results
    logger.info("\n" + "=" * 50)
    logger.info("STEP 5: CALCULATING RESULTS")
    logger.info("=" * 50)

    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)

    # Calculate accuracies for each condition
    conditions = ["standard", "triad", "sentence", "paragraph", "global"]
    accuracies = {}

    for condition in conditions:
        if f"{condition}_correct" in df_results.columns:
            accuracies[condition] = df_results[f"{condition}_correct"].mean()
        else:
            accuracies[condition] = 0.0

    # Display results (reproducing paper's format)
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS (n={len(df_results)}) - Reproducing Paper's J-Curve")
    logger.info("=" * 70)
    logger.info(f"Standard (Original):      {accuracies['standard']:.1%}")
    logger.info(f"Triad Shuffle (Local):    {accuracies['triad']:.1%}")
    logger.info(f"Sentence Shuffle:         {accuracies['sentence']:.1%}")
    logger.info(f"Paragraph Shuffle:        {accuracies['paragraph']:.1%}")
    logger.info(f"Global Shuffle (NIAH):    {accuracies['global']:.1%}")
    logger.info("=" * 70)

    # Key findings (J-curve analysis)
    j_curve_drop = accuracies["standard"] - accuracies["triad"]
    j_curve_rise = accuracies["global"] - accuracies["triad"]
    global_vs_standard = accuracies["global"] - accuracies["standard"]

    logger.info(f"\nJ-Curve Analysis:")
    logger.info(f"- Drop from Standard to Local: {j_curve_drop:.1%}")
    logger.info(f"- Rise from Local to Global:   {j_curve_rise:.1%}")
    logger.info(f"- Global vs Standard gap:      {global_vs_standard:.1%}")

    if j_curve_rise > 0.05:  # 5% threshold
        logger.info(
            "✓ J-CURVE DETECTED: Model performs better on globally shuffled text!"
        )
        logger.info("  This suggests reliance on 'reading' rather than pure 'recall'.")
    else:
        logger.info("⚠ No clear J-curve detected. Model may be failing on this task.")

    # Step 6: Save results
    logger.info("\n" + "=" * 50)
    logger.info("STEP 6: SAVING RESULTS")
    logger.info("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = output_dir / f"evaluation_results_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    logger.info(f"Detailed results saved to: {results_file}")

    # Save summary
    summary = {
        "experiment_info": {
            "timestamp": timestamp,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "year": args.year,
            "num_documents": len(df_results),
            "avg_document_length": (
                df_results["word_count"].mean() if "word_count" in df_results else None
            ),
            "methodology": "naturally_occurring_single_words",
        },
        "accuracies": accuracies,
        "j_curve_analysis": {
            "drop_standard_to_local": j_curve_drop,
            "rise_local_to_global": j_curve_rise,
            "global_vs_standard": global_vs_standard,
            "j_curve_detected": j_curve_rise > 0.05,
        },
        "model_info": model_info,
    }

    summary_file = output_dir / f"experiment_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    main()
