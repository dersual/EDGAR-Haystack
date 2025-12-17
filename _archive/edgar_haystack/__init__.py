"""
EDGAR-Haystack: Reproducing "Beyond the Haystack" with Financial Documents

A research framework for evaluating LLM retrieval capabilities on long-form
financial documents using domain-specific needles from SEC 10-K filings.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from edgar_haystack.data import EDGARLoader
from edgar_haystack.needles import NeedleExtractor, NeedleInserter
from edgar_haystack.evaluation import TextShuffler, PromptGenerator, ResponseScorer
from edgar_haystack.models import LlamaModel

__all__ = [
    "EDGARLoader",
    "NeedleExtractor",
    "NeedleInserter",
    "TextShuffler",
    "PromptGenerator",
    "ResponseScorer",
    "LlamaModel",
]
