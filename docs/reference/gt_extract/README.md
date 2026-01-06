# Ground Truth Extraction Pipeline

LLM-based ground truth extraction for EDGAR 10-K filings.

## Why LLM Instead of Regex?

The original regex-based extraction (`extract_edgar_ground_truth.ipynb`) had issues:
- Captured month names as states (e.g., "June" from "incorporated in June 1993")
- Captured company names instead of states
- High false positive rate

This LLM-based approach:
- Uses Llama-3.1-8B-Instruct to understand context
- Asks specific extraction questions
- Validates answers against known constraints (US states, year ranges, etc.)

## Usage

```bash
# Extract 500 samples (default)
python extract_ground_truth.py

# Extract 100 samples for testing
python extract_ground_truth.py --samples 100 --output test_gt.csv

# Extract all 500 for final ground truth
python extract_ground_truth.py --samples 500 --output edgar_ground_truth_llm.csv
```

## Output Format

CSV with columns:
- `cik`: Company identifier
- `year`: Filing year
- `filename`: Source filename (e.g., `92116_1993.txt`)
- `incorporation_state`: State where incorporated (validated against US_STATES)
- `incorporation_year`: Year of incorporation (validated 1800-2025)
- `employee_count`: Number of employees
- `fiscal_year_end`: When fiscal year ends
- `headquarters_state`: State where HQ is located (validated)
- `company_product`: Main product/service
- `ceo_lastname`: CEO's last name

## Validation

Each field has specific validation:
- **States**: Must match US_STATES list (handles abbreviations like "NY" → "New York")
- **Years**: Must be 4-digit year between 1800-2025
- **Numbers**: Extracts digits, removes commas
- **Names**: Must be capitalized, reasonable length

Invalid responses return `None` (empty in CSV).


