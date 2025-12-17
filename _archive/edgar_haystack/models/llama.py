"""
Model interface for Llama and other instruction-tuned models.

Provides a clean interface for querying models with consistent formatting
and response handling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LlamaModel:
    """
    Interface for Llama instruction-tuned models.

    Handles tokenization, generation, and response extraction with consistent settings
    optimized for the needle-in-haystack evaluation task.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map: str = "auto",
    ):
        """
        Initialize Llama model.

        Args:
            model_name: HuggingFace model identifier
            device_map: Device mapping strategy ("auto", "cpu", etc.)
        """
        self.model_name = model_name
        self.device_map = device_map

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency on modern GPUs
            device_map=device_map,
            trust_remote_code=True,
        )

        self.model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate response from chat messages.

        Args:
            messages: List of chat messages [{"role": "system/user", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            do_sample: Whether to use sampling

        Returns:
            Generated response string
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Extract only the generated part (exclude input)
        input_length = model_inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]

        # Decode response
        response = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        return response.strip()

    def query_single_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> str:
        """
        Convenience method for single prompt queries.

        Args:
            system_prompt: System message
            user_prompt: User message
            **kwargs: Additional generation parameters

        Returns:
            Model response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.generate_response(messages, **kwargs)

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text without actual tokenization.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English text
        return len(text) // 4

    def check_context_length(self, text: str, max_context: int = 8192) -> bool:
        """
        Check if text fits within model's context window.

        Args:
            text: Input text
            max_context: Maximum context length in tokens

        Returns:
            Whether text fits in context
        """
        estimated_tokens = self.estimate_token_count(text)
        return estimated_tokens <= max_context

    def truncate_text_to_context(
        self, text: str, max_tokens: int = 6000, preserve_start: int = 500
    ) -> str:
        """
        Truncate text to fit context window while preserving structure.

        Args:
            text: Input text
            max_tokens: Maximum tokens to keep
            preserve_start: Tokens to preserve from start (for context)

        Returns:
            Truncated text
        """
        words = text.split()
        estimated_tokens = len(words) * 1.3  # Rough word-to-token ratio

        if estimated_tokens <= max_tokens:
            return text

        # Calculate how many words to keep
        max_words = int(max_tokens / 1.3)
        preserve_start_words = int(preserve_start / 1.3)

        if max_words <= preserve_start_words:
            # If we can only keep the start, do that
            return " ".join(words[:max_words])

        # Keep start + end portions
        end_words = max_words - preserve_start_words
        truncated_words = words[:preserve_start_words] + words[-end_words:]

        return " ".join(truncated_words)

    def get_model_info(self) -> Dict:
        """
        Get model information for logging/debugging.

        Returns:
            Model info dictionary
        """
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(
                self.model.config, "max_position_embeddings", "unknown"
            ),
        }
