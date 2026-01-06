#!/usr/bin/env python3
"""
LLM-Based Ground Truth Extraction for EDGAR Corpus

This script uses Llama-3.1-8B-Instruct to extract ground truth answers from 
SEC 10-K filings. Much more accurate than pure regex extraction.

Extracts:
- incorporation_state: State where company was incorporated
- incorporation_year: Year of incorporation
- employee_count: Number of employees
- fiscal_year_end: When fiscal year ends
- headquarters_state: State where HQ is located
- company_product: Main product/service
- ceo_lastname: CEO's last name

Usage:
    python extract_ground_truth.py --samples 500 --output ground_truth.csv
"""

import os
import re
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_CONTEXT_CHARS = 20000  # 50K chars (~12K tokens) - fits in GPU memory
MAX_TOKENS = 14000  # Max tokens - safe limit for 24GB GPU
DEFAULT_SAMPLES = 500
DEFAULT_OUTPUT = "edgar_ground_truth_llm.csv"

# Valid US states for validation
US_STATES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'Puerto Rico', 'District of Columbia'
]

US_STATES_LOWER = [s.lower() for s in US_STATES]

# =============================================================================
# EXTRACTION PROMPTS
# =============================================================================

# Each extraction has: section to use, prompt template, validation function
EXTRACTIONS = {
    'incorporation_state': {
        'section': 'section_1',
        'prompt': """Read this SEC 10-K filing excerpt and answer the question.

Context:
{context}

Question: In which U.S. state was this company incorporated?

Instructions:
- Answer with ONLY the state name (e.g., "Delaware", "California")
- If the state is not mentioned, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'state',
    },
    'incorporation_year': {
        'section': 'section_1',
        'prompt': """Read this SEC 10-K filing excerpt and answer the question.

Context:
{context}

Question: In what year was this company incorporated or organized?

Instructions:
- Answer with ONLY the 4-digit year (e.g., "1985", "1993")
- If the year is not mentioned, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'year',
    },
    'employee_count': {
        'section': 'section_1',
        'prompt': """Read this SEC 10-K filing excerpt and answer the question.

Context:
{context}

Question: How many full-time employees does this company have?

Instructions:
- Answer with ONLY the number (e.g., "5000", "12500")
- Use digits only, no commas (e.g., "5000" not "5,000")
- If not mentioned, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'number',
    },
    'fiscal_year_end': {
        'section': 'section_1',
        'prompt': """Read this SEC 10-K filing excerpt and answer the question.

Context:
{context}

Question: On what date does the company's fiscal year end?

Instructions:
- Answer with the date (e.g., "December 31", "June 30", "March 31")
- If not mentioned, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'date',
    },
    'headquarters_state': {
        'section': 'section_2',
        'prompt': """Read this SEC 10-K filing excerpt about company properties and answer the question.

Context:
{context}

Question: In which U.S. state is the company's headquarters or principal executive offices located?

Instructions:
- Answer with ONLY the state name (e.g., "New York", "Texas")
- If the state is not mentioned, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'state',
    },
    'company_product': {
        'section': 'section_1',
        'prompt': """Read this SEC 10-K filing excerpt and answer the question.

Context:
{context}

Question: What is the main product, service, or business activity of this company?

Instructions:
- Answer in 2-5 words maximum (e.g., "telecommunications services", "oil and gas exploration")
- If not clear, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'text',
    },
    'ceo_lastname': {
        'section': 'section_10',
        'prompt': """Read this SEC 10-K filing excerpt about directors and officers and answer the question.

Context:
{context}

Question: What is the last name of the Chief Executive Officer (CEO)?

Instructions:
- Answer with ONLY the last name (e.g., "Smith", "Johnson")
- If not mentioned, answer "NOT_FOUND"
- Do not include any explanation

Answer:""",
        'validate': 'name',
    },
}

# =============================================================================
# MODEL LOADING
# =============================================================================

_model = None
_tokenizer = None


def load_model():
    """Load the LLM model and tokenizer."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print(f"Loading model: {MODEL_NAME}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    
    print("Model loaded successfully!")
    return _model, _tokenizer


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_state(answer):
    """Validate and normalize state name."""
    if not answer or answer.upper() == "NOT_FOUND":
        return None
    
    answer_clean = answer.strip().strip('."\'')
    answer_lower = answer_clean.lower()
    
    # Direct match
    for i, state_lower in enumerate(US_STATES_LOWER):
        if answer_lower == state_lower:
            return US_STATES[i]
    
    # Partial match (e.g., "State of Delaware" -> "Delaware")
    for i, state_lower in enumerate(US_STATES_LOWER):
        if state_lower in answer_lower:
            return US_STATES[i]
    
    # Common abbreviations
    abbrevs = {
        'ny': 'New York', 'ca': 'California', 'tx': 'Texas',
        'fl': 'Florida', 'pa': 'Pennsylvania', 'il': 'Illinois',
        'oh': 'Ohio', 'ga': 'Georgia', 'nc': 'North Carolina',
        'nj': 'New Jersey', 'va': 'Virginia', 'wa': 'Washington',
        'ma': 'Massachusetts', 'az': 'Arizona', 'tn': 'Tennessee',
        'mo': 'Missouri', 'md': 'Maryland', 'wi': 'Wisconsin',
        'mn': 'Minnesota', 'co': 'Colorado', 'al': 'Alabama',
        'sc': 'South Carolina', 'la': 'Louisiana', 'ky': 'Kentucky',
        'or': 'Oregon', 'ok': 'Oklahoma', 'ct': 'Connecticut',
        'ia': 'Iowa', 'ms': 'Mississippi', 'ar': 'Arkansas',
        'ut': 'Utah', 'nv': 'Nevada', 'ks': 'Kansas', 'nm': 'New Mexico',
        'ne': 'Nebraska', 'wv': 'West Virginia', 'id': 'Idaho',
        'hi': 'Hawaii', 'me': 'Maine', 'nh': 'New Hampshire',
        'ri': 'Rhode Island', 'mt': 'Montana', 'de': 'Delaware',
        'sd': 'South Dakota', 'nd': 'North Dakota', 'ak': 'Alaska',
        'vt': 'Vermont', 'wy': 'Wyoming', 'dc': 'District of Columbia',
    }
    if answer_lower in abbrevs:
        return abbrevs[answer_lower]
    
    return None


def validate_year(answer):
    """Validate year is reasonable."""
    if not answer or answer.upper() == "NOT_FOUND":
        return None
    
    # Extract 4-digit year
    match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', answer)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2025:
            return str(year)
    return None


def validate_number(answer):
    """Validate and clean employee count."""
    if not answer or answer.upper() == "NOT_FOUND":
        return None
    
    # Remove commas and extract number
    clean = answer.replace(',', '').replace(' ', '')
    match = re.search(r'(\d+)', clean)
    if match:
        num = int(match.group(1))
        if num > 0:
            return str(num)
    return None


def validate_date(answer):
    """Validate fiscal year end date."""
    if not answer or answer.upper() == "NOT_FOUND":
        return None
    
    answer_clean = answer.strip().strip('."\'')
    
    # Look for month + day pattern
    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    
    answer_lower = answer_clean.lower()
    for month in months:
        if month in answer_lower:
            # Try to extract day
            match = re.search(r'(\d{1,2})', answer_clean)
            if match:
                day = match.group(1)
                return f"{month.capitalize()} {day}"
            else:
                return month.capitalize()
    
    return answer_clean if len(answer_clean) < 20 else None


def validate_text(answer):
    """Validate product/service description."""
    if not answer or answer.upper() == "NOT_FOUND":
        return None
    
    answer_clean = answer.strip().strip('."\'')
    
    # Truncate if too long
    if len(answer_clean) > 100:
        answer_clean = answer_clean[:100]
    
    return answer_clean if len(answer_clean) > 2 else None


def validate_name(answer):
    """Validate CEO last name."""
    if not answer or answer.upper() == "NOT_FOUND":
        return None
    
    answer_clean = answer.strip().strip('."\'')
    
    # Should be a single word (last name)
    words = answer_clean.split()
    if words:
        # Take the last word if multiple
        name = words[-1]
        # Basic validation: starts with capital, reasonable length
        if name[0].isupper() and 2 <= len(name) <= 30:
            return name
    
    return None


VALIDATORS = {
    'state': validate_state,
    'year': validate_year,
    'number': validate_number,
    'date': validate_date,
    'text': validate_text,
    'name': validate_name,
}


# =============================================================================
# LLM EXTRACTION
# =============================================================================

def extract_with_llm(context, prompt_template, max_new_tokens=30):
    """Use LLM to extract information from context."""
    model, tokenizer = load_model()
    
    # Only truncate if we have a limit set (None = no limit)
    if MAX_CONTEXT_CHARS and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "..."
    
    prompt = prompt_template.format(context=context)
    
    # Llama 3.1 8B has 128K context - we use up to MAX_TOKENS
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean up response - take first line
    response = response.strip().split('\n')[0].strip()
    
    # Clear CUDA cache to prevent memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_all_fields(record, verbose=False):
    """Extract all ground truth fields from a single EDGAR record."""
    results = {
        'cik': record.get('cik', ''),
        'year': record.get('year', ''),
        'filename': record.get('filename', ''),
    }
    
    if verbose:
        print(f"\nExtracting {record.get('filename', 'unknown')}...")
    
    for field_name, config in EXTRACTIONS.items():
        section = config['section']
        context = record.get(section, '')
        
        if not context or len(context) < 50:
            results[field_name] = None
            if verbose:
                print(f"  {field_name}: {section} (EMPTY or too short) → None")
            continue
        
        try:
            # Extract with LLM
            raw_answer = extract_with_llm(context, config['prompt'])
            
            # Validate
            validator = VALIDATORS[config['validate']]
            validated = validator(raw_answer)
            
            results[field_name] = validated
            
            if verbose:
                section_len = len(context)
                raw_short = raw_answer[:30] + "..." if len(raw_answer) > 30 else raw_answer
                print(f"  {field_name}: {section} ({section_len:,} chars) → raw='{raw_short}' → validated='{validated}'")
                
        except Exception as e:
            print(f"  Error extracting {field_name}: {e}")
            results[field_name] = None
    
    return results


def load_edgar_corpus(num_samples):
    """Load EDGAR corpus samples."""
    print(f"Loading EDGAR corpus ({num_samples} samples)...")
    
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        split="train",
        streaming=True,
    )
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(item)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def run_extraction(num_samples, output_file, verbose=False):
    """Run the full extraction pipeline."""
    print("=" * 70)
    print("LLM-BASED GROUND TRUTH EXTRACTION")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Samples: {num_samples}")
    print(f"Output: {output_file}")
    print(f"Verbose: {verbose}")
    print()
    
    # Print section mapping
    print("Section mapping:")
    for field, config in EXTRACTIONS.items():
        print(f"  {field:<25} → {config['section']}")
    print()
    
    # Load model first
    load_model()
    
    # Load EDGAR corpus
    samples = load_edgar_corpus(num_samples)
    
    # Extract ground truth for each sample
    results = []
    
    print(f"\nExtracting ground truth from {len(samples)} documents...")
    iterator = samples if verbose else tqdm(samples, desc="Extracting")
    for sample in iterator:
        extracted = extract_all_fields(sample, verbose=verbose)
        results.append(extracted)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to match expected format
    column_order = [
        'cik', 'year', 'filename',
        'incorporation_state', 'incorporation_year', 'employee_count',
        'fiscal_year_end', 'headquarters_state', 'company_product',
        'ceo_lastname'
    ]
    df = df[[c for c in column_order if c in df.columns]]
    
    # Save to CSV - use "NULL" for missing values instead of empty
    df.fillna("NULL").to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} records to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    for col in column_order[3:]:  # Skip cik, year, filename
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {col:<25} {non_null:>5} / {len(df)} ({pct:>5.1f}%)")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based ground truth extraction from EDGAR corpus")
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help=f'Number of samples to process (default: {DEFAULT_SAMPLES})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output CSV file (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed extraction info for each document')
    
    args = parser.parse_args()
    
    # Run extraction
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    df = run_extraction(args.samples, output_path, verbose=args.verbose)
    
    print("\nDone!")

