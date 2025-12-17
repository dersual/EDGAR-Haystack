"""
Text shuffling for "Beyond the Haystack" evaluation.

Implements the paper's core manipulation: shuffle text at different granularities
to test whether models rely on "reading" (context coherence) vs "recall" (position).

Key insight: Models perform differently when local vs global coherence is disrupted.
"""

import random
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextShuffler:
    """
    Implement the paper's text shuffling methodology.

    The core finding: Models show J-shaped performance curve as shuffle window increases:
    - Standard (no shuffle): High performance
    - Local shuffle (triads): Lower performance (disrupts reading)
    - Global shuffle: Higher performance again (forces pure recall)
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize shuffler with consistent random seed.

        Args:
            random_seed: Seed for reproducible shuffling
        """
        self.random_seed = random_seed

    def shuffle_by_window(self, words: List[str], window_size: int) -> List[str]:
        """
        Shuffle words within sliding windows.

        Args:
            words: List of words from document
            window_size: Size of shuffle window
                - 1: No shuffle (standard condition)
                - 3: Triad shuffle (paper's local condition)
                - len(words): Global shuffle (NIAH condition)

        Returns:
            Shuffled word list
        """
        random.seed(self.random_seed)

        if window_size >= len(words):
            # Global shuffle (NIAH condition)
            shuffled = words.copy()
            random.shuffle(shuffled)
            return shuffled

        if window_size <= 1:
            # No shuffle (standard condition)
            return words.copy()

        # Local window shuffle
        shuffled = []
        for i in range(0, len(words), window_size):
            window = words[i : i + window_size]
            random.shuffle(window)
            shuffled.extend(window)

        return shuffled

    def create_shuffle_conditions(self, text: str) -> Dict[str, str]:
        """
        Create all shuffle conditions for the same document.

        This is the paper's key methodology: test the SAME document
        under multiple shuffle conditions to isolate the effect of coherence.

        Args:
            text: Input document text

        Returns:
            Dictionary mapping condition names to shuffled texts
        """
        words = text.split()

        # Apply different shuffle windows
        conditions = {
            "standard": " ".join(self.shuffle_by_window(words, 1)),  # No shuffle
            "triad": " ".join(
                self.shuffle_by_window(words, 3)
            ),  # Local shuffle (3 words)
            "sentence": " ".join(self.shuffle_by_window(words, 15)),  # ~Sentence level
            "paragraph": " ".join(
                self.shuffle_by_window(words, 50)
            ),  # ~Paragraph level
            "global": " ".join(
                self.shuffle_by_window(words, len(words))
            ),  # Full shuffle (NIAH)
        }

        logger.debug(
            f"Created {len(conditions)} shuffle conditions for {len(words)}-word document"
        )
        return conditions

    def validate_shuffling(self, original: str, shuffled: Dict[str, str]) -> Dict:
        """
        Validate that shuffling preserves word content while changing order.

        Args:
            original: Original text
            shuffled: Dictionary of shuffled versions

        Returns:
            Validation results
        """
        original_words = set(original.lower().split())

        validation = {}
        for condition, text in shuffled.items():
            shuffled_words = set(text.lower().split())

            validation[condition] = {
                "word_count_match": len(original.split()) == len(text.split()),
                "vocabulary_preserved": original_words == shuffled_words,
                "order_changed": original != text,
                "valid": True,
            }

            # Overall validity check
            validation[condition]["valid"] = (
                validation[condition]["word_count_match"]
                and validation[condition]["vocabulary_preserved"]
            )

        return validation


class PromptGenerator:
    """
    Generate evaluation prompts following "Beyond the Haystack" methodology.

    Key insight: The paper uses GENERIC prompts asking for "a word that appears
    exactly once" - no domain-specific hints or semantic guidance. This forces
    the model to rely on statistical properties rather than semantic understanding.
    """

    def __init__(self):
        # Generic prompt template following the paper's methodology
        self.system_prompt = "You are a helpful assistant. Answer concisely with just the requested word."
        self.user_template = (
            "In the following text, identify a word that appears exactly once. "
            "Respond with only that word.\n\n"
            "Text: {text}\n\n"
            "Word that appears once:"
        )

    def generate_prompt(self, text: str) -> Dict[str, str]:
        """
        Generate the standard prompt for needle retrieval.

        Args:
            text: Document text (potentially shuffled)

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        return {
            "system": self.system_prompt,
            "user": self.user_template.format(text=text),
        }

    def create_chat_messages(self, text: str) -> List[Dict[str, str]]:
        """
        Create chat format messages for model APIs.

        Args:
            text: Document text

        Returns:
            List of chat messages in standard format
        """
        prompt = self.generate_prompt(text)

        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]


class ResponseScorer:
    """
    Score model responses against ground truth needles.

    For the paper's methodology, this means checking if the model correctly
    identified the single-occurrence word.
    """

    def __init__(self):
        pass

    def extract_answer(self, response: str) -> str:
        """
        Extract the core answer from model response.

        Args:
            response: Raw model output

        Returns:
            Cleaned answer string (should be the single word)
        """
        # Basic cleaning
        response = response.strip()

        # Extract first meaningful part (models often add extra text)
        lines = response.split("\n")
        first_line = lines[0].strip()

        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "the answer is",
            "the word is",
            "answer:",
            "word:",
        ]

        answer = first_line.lower()
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix) :].strip()
                break

        # Extract just the first word (should be the needle)
        words = answer.split()
        if words:
            return words[0].strip()

        return answer.strip()

    def check_answer_match(self, response: str, target: str) -> Dict:
        """
        Check if response matches target answer.

        Args:
            response: Model response
            target: Ground truth needle word

        Returns:
            Match result dictionary
        """
        extracted_answer = self.extract_answer(response)
        target_clean = target.lower().strip()

        # Exact match (primary criterion)
        exact_match = extracted_answer == target_clean

        # Contains match (more lenient - sometimes model adds prefixes)
        contains_match = (
            target_clean in extracted_answer.lower() or extracted_answer in target_clean
        )

        return {
            "exact_match": exact_match,
            "contains_match": contains_match,
            "is_correct": exact_match,  # For single words, exact match is the standard
            "extracted_answer": extracted_answer,
            "target_answer": target_clean,
            "raw_response": response,
        }
        target_clean = target.lower().strip()

        # Exact match
        exact_match = extracted_answer == target_clean

        # Contains match (more lenient)
        contains_match = (
            target_clean in extracted_answer or extracted_answer in target_clean
        )

        # For specific needle types, apply specialized matching
        specialized_match = self._specialized_match(
            extracted_answer, target_clean, needle_type
        )

        return {
            "exact_match": exact_match,
            "contains_match": contains_match,
            "specialized_match": specialized_match,
            "is_correct": exact_match or specialized_match,
            "extracted_answer": extracted_answer,
            "target_answer": target_clean,
            "raw_response": response,
        }

    def _specialized_match(self, answer: str, target: str, needle_type: str) -> bool:
        """
        Apply needle-type specific matching logic.

        Args:
            answer: Extracted answer from response
            target: Target answer
            needle_type: Type of needle

        Returns:
            Whether answers match according to specialized rules
        """
        if needle_type == "auditor":
            # Match main firm name (ignore LLP, LLC suffixes)
            answer_firm = (
                answer.replace(" llp", "").replace(" llc", "").replace(" pc", "")
            )
            target_firm = (
                target.replace(" llp", "").replace(" llc", "").replace(" pc", "")
            )

            # Check if core firm name matches
            return answer_firm in target_firm or target_firm in answer_firm

        elif needle_type == "financial_figure":
            # Extract numeric values and compare
            import re

            answer_numbers = re.findall(r"[\d,]+\.?\d*", answer)
            target_numbers = re.findall(r"[\d,]+\.?\d*", target)

            if answer_numbers and target_numbers:
                return answer_numbers[0].replace(",", "") == target_numbers[0].replace(
                    ",", ""
                )

        elif needle_type == "executive":
            # Match last name at minimum
            answer_words = answer.split()
            target_words = target.split()

            if answer_words and target_words:
                # Check if last names match
                return answer_words[-1] == target_words[-1]

        # Default: substring matching
        return target in answer or answer in target

    def calculate_accuracy(self, results: List[Dict]) -> Dict:
        """
        Calculate accuracy metrics across multiple results.

        Args:
            results: List of individual match results

        Returns:
            Aggregate accuracy metrics
        """
        if not results:
            return {"exact": 0.0, "contains": 0.0, "specialized": 0.0, "overall": 0.0}

        exact_correct = sum(1 for r in results if r["exact_match"])
        contains_correct = sum(1 for r in results if r["contains_match"])
        specialized_correct = sum(1 for r in results if r["specialized_match"])
        overall_correct = sum(1 for r in results if r["is_correct"])

        total = len(results)

        return {
            "exact": exact_correct / total,
            "contains": contains_correct / total,
            "specialized": specialized_correct / total,
            "overall": overall_correct / total,
            "total_samples": total,
        }
