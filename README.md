# EDGAR Haystack Experiments

Recreating "Beyond the Haystack" paper findings using SEC 10-K filings and Llama-3-8B-Instruct on Lambda Cloud GPUs.

## What This Is

Testing how small models retrieve information from financial documents under different shuffling conditions to find interesting patterns.

**Core Question**: Do models perform differently on shuffled vs unshuffled financial text? Which question types show interesting patterns?

## Current Status

### ✅ Working
- `notebooks/Beyond_The_Haystack_Recreation_Using_Edgar_V2.ipynb` - Baseline experiment with single-instance word retrieval
- Shows J-curve pattern: models do better on globally shuffled text
- Uses `c3po-ai/edgar-corpus` dataset (10-K filings)

### 🚧 Next Steps
1. Automate question/answer generation for different query types
2. Extract answers using simple regex patterns (like iahd repo does)
3. Run batch experiments across multiple documents
4. Analyze which question types show shuffle sensitivity

## Project Structure

```
EDGAR-Haystack/
├── notebooks/
│   ├── Beyond_The_Haystack_Recreation_Using_Edgar_V2.ipynb  # Current working baseline
│   └── [new notebooks as needed]
│
├── requirements.txt                  # Dependencies
├── Beyond_Haystack_RS_Paper.pdf     # Original paper
└── README.md                         # This file

# Legacy/Unused (can ignore or delete)
├── edgar_haystack/                   # Old package structure - not needed
├── scripts/                          # Old automation - not needed  
├── configs/                          # Old configs - not needed
└── *.md files                        # Over-engineered docs - ignore
```

## Quick Start on Lambda Cloud

### 1. Launch GPU Instance
```bash
export LAMBDA_API_KEY="your_key_here"

curl --user "${LAMBDA_API_KEY}:" \
  --request POST "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
  --data '{
    "region_name":"us-west-1",
    "instance_type_name":"gpu_1x_a10",
    "ssh_key_names":["your-ssh-key"],
    "name":"edgar-experiment"
  }'
```

### 2. Get Instance URL
```bash
# Save instance ID from previous command, then:
curl --user "${LAMBDA_API_KEY}:" \
  --request GET "https://cloud.lambda.ai/api/v1/instances/<INSTANCE_ID>"
```

### 3. Open Jupyter and Run Notebook
- Open the Jupyter URL from the response
- Upload `Beyond_The_Haystack_Recreation_Using_Edgar_V2.ipynb`
- Run cells

## The Experiment

### What We're Testing
1. Load 10-K financial filings from HuggingFace
2. Create different shuffle conditions (no shuffle → local → global)
3. Ask questions about the documents
4. Compare accuracy across conditions

### Question Types to Test
- **Single-instance words** (already working)
- **Financial facts**: "What was the total revenue?"
- **Entity names**: "Who is the auditor?"
- **Dates**: "What is the fiscal year end?"
- **Numbers**: "What were total assets?"

### Answer Extraction (Simple Approach)
Following the iahd repo pattern - use regex to extract answers:

```python
def extract_answer(model_response, answer_type):
    """Simple regex-based answer extraction"""
    if answer_type == 'word':
        return re.search(r'\b\w+\b', model_response).group(0)
    elif answer_type == 'number':
        return re.search(r'\$?[\d,]+\.?\d*', model_response).group(0)
    elif answer_type == 'date':
        return re.search(r'\d{1,2}/\d{1,2}/\d{4}', model_response).group(0)
    # etc...
```

## Next Implementation Steps

### Step 1: Begin using LLama-3-8B-Instruct and Lambda Cloud 
Switch from Qwen-7B to Meta's Llama-3-8B-Instruct model hosted on Lambda Cloud A10 GPU for better performance and cost efficiency. 

### Step 2: Question Template System
Create a simple function that generates questions from documents:

```python
def generate_questions(document_text):
    """Generate different question types from a document"""
    questions = []
    
    # Single instance word (already working)
    word = find_single_occurrence_word(document_text)
    questions.append({
        'type': 'single_word',
        'question': f'Find a word that appears exactly once in the text.',
        'answer': word
    })
    
    # Extract financial facts with regex
    revenue = re.search(r'revenue.*?\$([0-9,]+)', document_text, re.I)
    if revenue:
        questions.append({
            'type': 'financial_fact',
            'question': 'What was the total revenue?',
            'answer': revenue.group(1)
        })
    
    # Add more question types...
    return questions
```

### Step 3: Batch Processing
Run multiple documents and save results:

```python
results = []
for doc in documents[:50]:  # Test on 50 docs
    questions = generate_questions(doc)
    for q in questions:
        for shuffle_type in ['standard', 'local', 'global']:
            shuffled_text = apply_shuffle(doc, shuffle_type)
            response = query_model(shuffled_text, q['question'])
            answer = extract_answer(response, q['type'])
            
            results.append({
                'doc_id': doc['id'],
                'question_type': q['type'],
                'shuffle': shuffle_type,
                'correct': (answer == q['answer'])
            })

# Save to CSV
pd.DataFrame(results).to_csv('results.csv')
```

### Step 3: Analyze Patterns
Simple analysis of which question types show interesting shuffle effects:

```python
df = pd.read_csv('results.csv')
accuracy_by_type = df.groupby(['question_type', 'shuffle'])['correct'].mean()
print(accuracy_by_type)
```

## Dataset & Model

- **Dataset**: `c3po-ai/edgar-corpus` (SEC 10-K Item 7 sections)
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Compute**: Lambda Cloud A10 GPU ($0.75/hour)

## Key Insight from Paper

The paper found models do **better** on globally shuffled (incoherent) text for certain tasks. This suggests they're not really "reading" but pattern matching. We want to see if this holds for financial documents and which question types show this pattern.

## References

- Paper: https://aclanthology.org/2025.nllp-1.5.pdf
- Dataset: https://huggingface.co/datasets/c3po-ai/edgar-corpus
- Answer extraction approach: https://github.com/harryila/iahd