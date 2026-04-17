import csv
import os
from get_judge_lm import llm_inference
import pandas as pd

# The prompt template for the baseline
PROMPT_TEMPLATE = '''You have to logically think and answer the question presented. 
Provide a clear YES, NO, or INCONCLUSIVE at the start of your response.

Question: {question}
'''

def parse_verdict(response):
    """Simple parser to extract yes/no for the 'Prediction' column."""
    response_clean = response.lower().strip()
    if response_clean.startswith('yes'): return 'yes'
    if response_clean.startswith('no'): return 'no'
    return 'inconclusive'

def run_and_log_baseline(input_csv, log_csv):
    dataset = pd.read_csv(input_csv)
    
    # Check if file exists to write header
    file_exists = os.path.isfile(log_csv)
    
    with open(log_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header according to your requested format
        # Index, Question, Ground Truth, Prediction, IsCorrect, Score1, Score2, Rationale
        if not file_exists:
            writer.writerow(['index', 'question', 'ground_truth', 'prediction', 'is_correct', 'score_1', 'score_2', 'rationale'])

        for i, row in dataset.iterrows():
            question = row['question']
            ground_truth = str(row['label']).lower().strip() # Assuming 'label' column exists
            
            # 1. Run Inference
            prompt = PROMPT_TEMPLATE.format(question=question)
            response = llm_inference(prompt=prompt, model_name="mistral-7b", temperature=0.7)
            
            # 2. Process Output
            prediction = parse_verdict(response)
            is_correct = (prediction == ground_truth)
            
            # 3. Placeholder for Scores (You can replace 9.0 with your FFN score if available)
            score_1 = 9.0 
            score_2 = 9.0
            rationale = response.replace('\n', ' ').strip()[:300] # Limit length for CSV readability

            # 4. Write Row
            writer.writerow([i, question, ground_truth, prediction, is_correct, score_1, score_2, rationale])
            print(f"Processed index {i}: Correct={is_correct}")

if __name__ == "__main__":
    input_csv = 'pubmedqa_train.csv'  # Update with your dataset path
    log_csv = 'debate_eval_results_mistral_baseline.csv'  # Output log file
    run_and_log_baseline(input_csv, log_csv)