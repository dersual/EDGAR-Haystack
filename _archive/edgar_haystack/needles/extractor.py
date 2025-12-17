"""
Needle extraction following "Beyond the Haystack" paper methodology.

This module finds naturally occurring single-instance words in documents,
exactly as described in the original paper. No artificial insertion,
no domain-specific targeting - just words that appear exactly once.

Key methodology from paper:
1. Find words that appear exactly once in the original document
2. These become the "needles" to search for
3. Test retrieval across different text shuffle conditions
4. Measure how coherence disruption affects retrieval
"""

import re
import random
from typing import List, Dict, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class NeedleExtractor:
    """
    Extract naturally occurring single-instance words from documents.

    This follows the exact methodology from "Beyond the Haystack" paper:
    - Find words that appear exactly once in the text
    - No artificial insertion or domain-specific targeting
    - Use generic prompting: "find a word that appears exactly once"
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the extractor.

        Args:
            random_seed: Random seed for reproducible needle selection
        """
        self.random_seed = random_seed

    def find_single_occurrence_words(
        self, text: str, min_length: int = 5, max_length: int = 12
    ) -> Optional[Dict]:
        """
        Find words that appear exactly once in the text.
        This matches the paper's approach of using naturally occurring needles.

        Args:
            text: Input document text
            min_length: Minimum word length to consider
            max_length: Maximum word length to consider

        Returns:
            Dictionary with needle info or None if no suitable needle found
        """
        if not text or len(text.strip()) < 100:
            return None

        # Normalize and extract words
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        # Basic quality check
        if len(words) < 200:  # Too short
            return None

        # For very long documents, limit to manageable size (matching your notebook)
        if len(words) > 512:
            words = words[:512]

        # Count word frequencies
        counts = Counter(words)

        # Find words that appear exactly once and meet criteria
        candidates = [
            word
            for word, count in counts.items()
            if count == 1
            and min_length <= len(word) <= max_length
            and word.isalpha()  # No numbers or special characters
            and not self._is_stopword(word)  # Filter out common stopwords
        ]

        if not candidates:
            return None

        # Set random seed for reproducible results
        random.seed(self.random_seed)

        # Pick a random candidate as the "needle"
        needle = random.choice(candidates)
        needle_position = words.index(needle)

        return {
            "words": words,
            "needle": needle,
            "position": needle_position,
            "text": " ".join(words),
            "total_words": len(words),
            "unique_words": len(set(words)),
            "candidates_found": len(candidates),
        }

    def _is_stopword(self, word: str) -> bool:
        """
        Check if word is a common stopword that should be filtered out.

        Args:
            word: Word to check

        Returns:
            True if word should be filtered out
        """
        # Common stopwords that appear in single occurrence but aren't meaningful needles
        stopwords = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "under",
            "over",
            "within",
            "without",
            "this",
            "that",
            "these",
            "those",
            "they",
            "them",
            "their",
            "there",
            "here",
            "where",
            "when",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "very",
            "too",
            "more",
            "most",
            "some",
            "any",
            "each",
            "every",
            "all",
            "both",
            "either",
            "neither",
            "none",
            "one",
            "two",
            "first",
            "last",
            "also",
            "just",
            "only",
            "still",
            "yet",
            "already",
            "again",
            "once",
            "then",
            "now",
            "soon",
            "later",
            "today",
            "tomorrow",
            "yesterday",
        }
        return word.lower() in stopwords

    def create_evaluation_case(self, text: str) -> Optional[Dict]:
        """
        Create a complete evaluation case from a document.

        Args:
            text: Document text

        Returns:
            Evaluation case dictionary or None if no needle found
        """
        needle_result = self.find_single_occurrence_words(text)

        if not needle_result:
            return None

        return {
            "original_text": text,
            "words": needle_result["words"],
            "needle": needle_result["needle"],
            "needle_position": needle_result["position"],
            "text_for_shuffling": needle_result["text"],
            "statistics": {
                "total_words": needle_result["total_words"],
                "unique_words": needle_result["unique_words"],
                "candidates_found": needle_result["candidates_found"],
            },
        }

    def create_evaluation_prompt(self, shuffled_text: str) -> str:
        """
        Create the evaluation prompt following the paper's methodology.

        This is the key difference from domain-specific approaches:
        - Generic prompt asking for "a word that appears exactly once"
        - No semantic hints about what to look for
        - Forces the model to rely on statistical properties

        Args:
            shuffled_text: Text (potentially shuffled) to search in

        Returns:
            Evaluation prompt string
        """
        prompt = (
            "In the following text, identify a word that appears exactly once. "
            "Respond with only that word.\n\n"
            f"Text: {shuffled_text}\n\n"
            "Word that appears once:"
        )
        return prompt

    def validate_text_quality(self, text: str) -> Dict:
        """
        Assess text quality for needle extraction.

        Args:
            text: Document text to assess

        Returns:
            Quality assessment dictionary
        """
        if not text:
            return {"suitable": False, "reason": "empty_text"}

        words = text.split()
        word_count = len(words)

        # Basic length checks
        if word_count < 200:
            return {"suitable": False, "reason": "too_short", "word_count": word_count}

        if word_count > 10000:
            return {"suitable": False, "reason": "too_long", "word_count": word_count}

        # Check if we can find any single-occurrence words
        needle_result = self.find_single_occurrence_words(text)

        if not needle_result:
            return {
                "suitable": False,
                "reason": "no_needles_found",
                "word_count": word_count,
            }

        return {
            "suitable": True,
            "word_count": word_count,
            "candidates_found": needle_result["candidates_found"],
            "sample_needle": needle_result["needle"],
        }


class NeedleInserter:
    """
    DEPRECATED: The paper does NOT use artificial needle insertion.

    This class is kept only for compatibility but should not be used
    in the main evaluation. The paper's methodology relies on naturally
    occurring single-instance words, not artificially inserted needles.
    """

    def __init__(self):
        logger.warning(
            "NeedleInserter is deprecated. The 'Beyond the Haystack' paper "
            "uses naturally occurring words, not artificial insertion. "
            "Use NeedleExtractor.find_single_occurrence_words() instead."
        )

    def insert_needle_at_position(
        self, text: str, needle: str, position_ratio: float = 0.5
    ):
        """Deprecated method - do not use."""
        raise NotImplementedError(
            "Artificial needle insertion is not part of the paper's methodology. "
            "Use NeedleExtractor.find_single_occurrence_words() instead."
        )
