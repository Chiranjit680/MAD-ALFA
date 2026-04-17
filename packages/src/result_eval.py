import pandas as pd

results = pd.read_csv('debate_eval_results_Mistral_multirebuttal.csv')
correct = 0
valid = 0

for _, row in results.iterrows():
    verdict = str(row['verdict']).strip().lower()
    if verdict in ['yes', 'no']:
        valid += 1
        is_correct = str(row['correct']).strip().lower() in ['true', '1', 'yes']
        if is_correct:
            correct += 1

if valid == 0:
    print('Accuracy: N/A (0/0)')
else:
    print(f'Accuracy: {correct/valid:.2%} ({correct}/{valid})')