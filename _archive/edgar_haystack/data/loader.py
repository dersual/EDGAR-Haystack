"""
Data loading and preprocessing for EDGAR corpus
"""

from datasets import load_dataset
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EDGARLoader:
    """
    Load and preprocess EDGAR 10-K filings from HuggingFace datasets.

    Handles both datasets:
    - c3po-ai/edgar-corpus (your current choice)
    - eloukas/edgar-corpus (alternative)
    """

    def __init__(self, dataset_name: str = "c3po-ai/edgar-corpus", year: str = "2020"):
        """
        Initialize EDGAR dataset loader.

        Args:
            dataset_name: HuggingFace dataset name
            year: Year of filings to load (2020, 2021, etc.)
        """
        self.dataset_name = dataset_name
        self.year = year
        self.train_dataset = None
        self.test_dataset = None

    def load_dataset(
        self, split: str = "test", limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Load EDGAR dataset from HuggingFace.

        Args:
            split: 'train' or 'test'
            limit: Maximum number of documents to load

        Returns:
            List of document dictionaries
        """
        if self.dataset_name == "c3po-ai/edgar-corpus":
            # Use parquet URLs (your current approach)
            if split == "train":
                url = f"https://huggingface.co/datasets/c3po-ai/edgar-corpus/resolve/refs%2Fconvert%2Fparquet/year_{self.year}/train/0000.parquet"
            else:
                url = f"https://huggingface.co/datasets/c3po-ai/edgar-corpus/resolve/refs%2Fconvert%2Fparquet/year_{self.year}/test/0000.parquet"

            dataset = load_dataset("parquet", data_files={split: url}, split=split)
        else:
            # Alternative: direct dataset loading
            dataset = load_dataset(self.dataset_name, split=split)

        # Convert to list and apply limit
        documents = list(dataset)
        if limit:
            documents = documents[:limit]

        logger.info(
            f"Loaded {len(documents)} documents from {self.dataset_name} ({split})"
        )
        return documents

    def extract_section_7(self, document: Dict) -> Optional[str]:
        """
        Extract Item 7 (Management's Discussion and Analysis) from 10-K filing.

        This section contains the most narrative content and is ideal for needle insertion.

        Args:
            document: Document dictionary from dataset

        Returns:
            Section 7 text or None if not found
        """
        # Try different field names depending on dataset structure
        for field_name in ["section_7", "item_7", "mda", "text"]:
            if field_name in document and document[field_name]:
                text = document[field_name]
                break
        else:
            return None

        # Basic quality checks
        if len(text.strip()) < 1000:  # Too short
            return None

        if len(text.split()) < 200:  # Too few words
            return None

        return text.strip()

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for evaluation.

        Args:
            text: Raw text from filing

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common SEC boilerplate that might interfere
        # (you can expand this based on what you see in the data)
        boilerplate_patterns = [
            r"TABLE OF CONTENTS.*?(?=\n\n|\. )",
            r"UNITED STATES.*?WASHINGTON.*?D\.C\..*?\n",
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        return text.strip()

    def get_documents_for_evaluation(self, num_docs: int = 100) -> List[Dict]:
        """
        Get clean documents ready for needle extraction and evaluation.

        Args:
            num_docs: Number of documents to return

        Returns:
            List of processed documents with section_7 text
        """
        logger.info(f"Loading documents for evaluation...")

        # Load test set (more stable for evaluation)
        raw_documents = self.load_dataset(
            split="test", limit=num_docs * 3
        )  # Get extra in case some are filtered

        processed_docs = []
        for doc in raw_documents:
            # Extract and clean section 7
            section_7 = self.extract_section_7(doc)
            if section_7:
                clean_text = self.preprocess_text(section_7)

                # Final quality check
                if len(clean_text.split()) >= 500:  # At least 500 words
                    processed_docs.append(
                        {
                            "text": clean_text,
                            "word_count": len(clean_text.split()),
                            "char_count": len(clean_text),
                            "source_doc": doc,  # Keep reference to original
                        }
                    )

            if len(processed_docs) >= num_docs:
                break

        logger.info(f"Successfully processed {len(processed_docs)} documents")
        logger.info(
            f"Average length: {sum(d['word_count'] for d in processed_docs) / len(processed_docs):.0f} words"
        )

        return processed_docs
