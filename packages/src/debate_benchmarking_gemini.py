import os

os.environ["CUDA_VISIBLE_DEVICES"]              = "5"
os.environ["CUDA_DEVICE_ORDER"]                 = "PCI_BUS_ID"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from debate_engine import run_debate
import pandas as pd
from dotenv import load_dotenv
from model_inference import load_models
from google import genai
import re

load_dotenv()

# ⚠️  SECURITY FIX — never hardcode API keys
# Move to .env: GEMINI_API_KEY=AIzaSy...
gemini_client = genai.Client(api_key="AIzaSyDzny21HlBRAMXMP4mJt6UXkdZ5MD5CMWo")
print("✅ Gemini client ready — using gemini-2.0-flash")

df = pd.read_csv("pubmedqa_train.csv")


def run_judge(
    ground_truth: str,
    final_answer: str,
    question    : str
) -> tuple[float, str]:

    prompt = f"""You are a strict medical reasoning judge.

Question        : {question}
Ground Truth    : {ground_truth}
Debate Verdict  : {final_answer}

Score the debate verdict out of 10 based on:
- Factual alignment with ground truth   (5 points)
- Logical reasoning quality             (5 points)

Reply in this EXACT format and nothing else:
SCORE: <number between 0 and 10>
REASON: <one sentence explanation>

Response:"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    decoded = response.text.strip()

    score_match  = re.search(r'SCORE:\s*([0-9]+(?:\.[0-9]+)?)', decoded)
    reason_match = re.search(r'REASON:\s*(.+)', decoded, re.IGNORECASE)

    score     = float(score_match.group(1))   if score_match  else 0.0
    score     = max(0.0, min(10.0, score))
    reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"

    return score, reasoning


def run_debate_on_pubmedqa():
    no_rows   = 10000
    correct   = 0
    total     = 0
    score_sum = 0.0
    results_log = []

    load_models("argument_quality_model_4features.pth")

    for index, row in df.head(no_rows).iterrows():
        question = row['question']
        answer   = row['long_answer']
        decision = row['final_decision']

        try:
            result = run_debate(
                topic=question,
                max_arg_iterations=5,
                hf_api_key=os.getenv("HF_TOKEN")
            )

            verdict  = result['verdict']['answer'].lower()
            expected = decision.lower()
            total   += 1

            score, reasoning = run_judge(
                ground_truth=answer,
                final_answer=verdict,
                question=question
            )

            score_sum += score
            avg_score  = score_sum / total
            is_correct = verdict == expected
            if is_correct:
                correct += 1

            status = "✅" if is_correct else "❌"
            print(
                f"{status} [{index:>4}] "
                f"Score: {score:>5.1f}/10 | "
                f"Avg: {avg_score:>5.2f}/10 | "
                f"Accuracy: {correct}/{total} | "
                f"Reason: {reasoning}"
            )

            results_log.append({
                "index"    : index,
                "question" : question,
                "expected" : expected,
                "verdict"  : verdict,
                "correct"  : is_correct,
                "score"    : score,
                "avg_score": avg_score,
                "reasoning": reasoning
            })

        except Exception as e:
            total += 1
            print(f"⚠️  Error on row {index}: {e}")
            results_log.append({
                "index"    : index,
                "question" : question,
                "expected" : decision.lower(),
                "verdict"  : "error",
                "correct"  : False,
                "score"    : 0.0,
                "avg_score": score_sum / total if total > 0 else 0.0,
                "reasoning": f"Error: {str(e)}"
            })

    final_avg = score_sum / total if total > 0 else 0.0
    accuracy  = correct   / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Total Evaluated : {total}")
    print(f"  Correct Verdicts: {correct}/{total} = {accuracy:.2%}")
    print(f"  Final Avg Score : {final_avg:.2f}/10")
    print(f"  Judge backend   : Gemini 2.0 Flash")
    print(f"{'='*60}")

    results_df = pd.DataFrame(results_log)
    results_df.to_csv("debate_eval_results.csv", index=False)

    print("\nScore Distribution:")
    bins   = [0, 2, 4, 6, 8, 10]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]
    results_df['score_band'] = pd.cut(
        results_df['score'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    print(results_df['score_band'].value_counts().sort_index().to_string())

    return final_avg, accuracy, results_log


if __name__ == "__main__":
    final_avg, accuracy, log = run_debate_on_pubmedqa()