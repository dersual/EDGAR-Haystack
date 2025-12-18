#!/usr/bin/env python3
"""
Compare LLM extraction vs your regex extraction from logit_lens_llama.py
Also check if missing data exists in the documents.
"""

import sys
sys.path.insert(0, '/home/ubuntu/iahd')

import re
from datasets import load_dataset

# Import your regex functions from logit_lens_llama.py
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
    'West Virginia', 'Wisconsin', 'Wyoming'
]

def extract_state_of_incorporation(text):
    """Your regex extraction from logit_lens_llama.py"""
    text_lower = text.lower()
    patterns = [
        r'incorporated (?:in|under the laws of) (?:the state of )?(\w+(?:\s+\w+)?)',
        r'a (\w+(?:\s+\w+)?) corporation',
        r'organized (?:in|under the laws of) (?:the state of )?(\w+(?:\s+\w+)?)',
        r'(?:the |a )(\w+(?:\s+\w+)?) company',
        r'state of incorporation[:\s]+(\w+(?:\s+\w+)?)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            for state in US_STATES:
                if state.lower() == match.lower() or state.lower() in match.lower():
                    return state
    return None

def extract_incorporation_year(text):
    """Your regex extraction from logit_lens_llama.py"""
    patterns = [
        r'incorporated (?:in |on |)(?:\w+ )?(\d{4})',
        r'organized (?:in |on |)(?:\w+ )?(\d{4})',
        r'founded (?:in |on |)(?:\w+ )?(\d{4})',
        r'established (?:in |on |)(?:\w+ )?(\d{4})',
        r'formed (?:in |on |)(?:\w+ )?(\d{4})',
        r'since (\d{4})',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            year = int(match)
            if 1800 <= year <= 2000:
                return str(year)
    return None

def extract_headquarters_state(text):
    """Your regex extraction from logit_lens_llama.py"""
    text_lower = text.lower()
    patterns = [
        r'(?:headquarters|principal (?:executive )?offices?|corporate offices?) (?:is |are |)(?:located |)in ([^,\.\n]+)',
        r'(?:located|headquartered) in ([^,\.\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            for state in US_STATES:
                if state.lower() in match.lower():
                    return state
    return None

def extract_ceo_name(text):
    """Your regex extraction from logit_lens_llama.py"""
    patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+)[,\s]+(?:is |serves as |)(?:the |our |)(?:Chief Executive Officer|CEO|President and CEO)',
        r'(?:Chief Executive Officer|CEO|President and CEO)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
        r'([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)[,\s]+(?:Chief Executive Officer|CEO)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            full_name = matches[0]
            parts = full_name.replace('.', '').split()
            if parts:
                return parts[-1]
    return None

def check_if_data_exists(text, field_type):
    """Check if the data might exist in the text even if extraction failed."""
    text_lower = text.lower()
    
    if field_type == 'employee':
        # Look for employee-related keywords
        patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*(?:full[- ]time\s*)?employees',
            r'employees.*?(\d{1,3}(?:,\d{3})*)',
            r'approximately\s*(\d{1,3}(?:,\d{3})*)',
            r'employed\s*(?:approximately\s*)?(\d{1,3}(?:,\d{3})*)',
        ]
        for p in patterns:
            match = re.search(p, text_lower)
            if match:
                return f"FOUND: '{match.group(0)[:60]}...'"
        return "NOT IN DOCUMENT (no employee mentions found)"
    
    elif field_type == 'fiscal_year':
        patterns = [
            r'fiscal year[s]?\s*(?:end|ending|ended)',
            r'year[s]?\s*(?:end|ending|ended)',
        ]
        for p in patterns:
            match = re.search(p, text_lower)
            if match:
                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text_lower), match.end() + 50)
                return f"FOUND: '...{text_lower[start:end]}...'"
        return "NOT IN DOCUMENT (no fiscal year mentions found)"
    
    elif field_type == 'ceo':
        patterns = [
            r'chief executive officer',
            r'\bceo\b',
            r'president and chief',
        ]
        for p in patterns:
            match = re.search(p, text_lower)
            if match:
                start = max(0, match.start() - 30)
                end = min(len(text_lower), match.end() + 50)
                return f"FOUND: '...{text_lower[start:end]}...'"
        return "NOT IN DOCUMENT (no CEO mentions found)"
    
    return "Unknown field type"


def main():
    print("Loading 5 samples from EDGAR corpus...")
    dataset = load_dataset("c3po-ai/edgar-corpus", split="train", streaming=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= 5:
            break
        samples.append(item)
    
    print("\n" + "=" * 80)
    print("COMPARISON: YOUR REGEX vs LLM EXTRACTION")
    print("=" * 80)
    
    # LLM results from previous run
    llm_results = {
        '92116_1993.txt': {'inc_state': 'California', 'inc_year': '1929', 'hq_state': 'California', 'employee': '2500', 'fiscal': 'December 31', 'ceo': None},
        '103730_1993.txt': {'inc_state': 'Delaware', 'inc_year': '1962', 'hq_state': 'Pennsylvania', 'employee': '12000', 'fiscal': None, 'ceo': None},
        '100240_1993.txt': {'inc_state': 'Georgia', 'inc_year': '1965', 'hq_state': 'Georgia', 'employee': None, 'fiscal': 'December 31', 'ceo': None},
        '58696_1993.txt': {'inc_state': 'Florida', 'inc_year': '1954', 'hq_state': 'Florida', 'employee': None, 'fiscal': None, 'ceo': None},
        '46207_1993.txt': {'inc_state': 'Hawaii', 'inc_year': '1891', 'hq_state': 'Hawaii', 'employee': None, 'fiscal': 'September 30', 'ceo': None},
    }
    
    for sample in samples:
        filename = sample.get('filename', 'unknown')
        section_1 = sample.get('section_1', '')
        section_2 = sample.get('section_2', '')
        section_10 = sample.get('section_10', '')
        
        print(f"\n{'─' * 80}")
        print(f"FILE: {filename}")
        print(f"  section_1: {len(section_1):,} chars")
        print(f"  section_2: {len(section_2):,} chars")
        print(f"  section_10: {len(section_10):,} chars")
        print()
        
        # Your regex extraction
        your_inc_state = extract_state_of_incorporation(section_1)
        your_inc_year = extract_incorporation_year(section_1)
        your_hq_state = extract_headquarters_state(section_2)
        your_ceo = extract_ceo_name(section_10)
        
        # LLM extraction
        llm = llm_results.get(filename, {})
        
        print(f"  {'Field':<25} {'YOUR REGEX':<20} {'LLM':<20} {'Match?'}")
        print(f"  {'-' * 70}")
        
        # Incorporation State
        match = "✓" if your_inc_state == llm.get('inc_state') else "✗"
        print(f"  {'incorporation_state':<25} {str(your_inc_state):<20} {str(llm.get('inc_state')):<20} {match}")
        
        # Incorporation Year
        match = "✓" if your_inc_year == llm.get('inc_year') else "✗"
        print(f"  {'incorporation_year':<25} {str(your_inc_year):<20} {str(llm.get('inc_year')):<20} {match}")
        
        # HQ State
        match = "✓" if your_hq_state == llm.get('hq_state') else "✗"
        print(f"  {'headquarters_state':<25} {str(your_hq_state):<20} {str(llm.get('hq_state')):<20} {match}")
        
        # CEO
        match = "✓" if your_ceo == llm.get('ceo') else ("~" if your_ceo is None and llm.get('ceo') is None else "✗")
        print(f"  {'ceo_lastname':<25} {str(your_ceo):<20} {str(llm.get('ceo')):<20} {match}")
        
        # Check missing data
        print()
        print(f"  MISSING DATA ANALYSIS:")
        
        if llm.get('employee') is None:
            status = check_if_data_exists(section_1, 'employee')
            print(f"    employee_count: {status}")
        
        if llm.get('fiscal') is None:
            status = check_if_data_exists(section_1, 'fiscal_year')
            print(f"    fiscal_year_end: {status}")
        
        if llm.get('ceo') is None:
            status = check_if_data_exists(section_10, 'ceo')
            print(f"    ceo_lastname: {status}")


if __name__ == "__main__":
    main()


