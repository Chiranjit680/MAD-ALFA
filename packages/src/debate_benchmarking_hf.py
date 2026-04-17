import os
import re
import argparse
import logging
from dotenv import load_dotenv
load_dotenv()
os.environ["CUDA_LAUNCH_BLOCKING"]              = "0"
os.environ["CUDA_VISIBLE_DEVICES"]              = "0"
os.environ["CUDA_DEVICE_ORDER"]                 = "PCI_BUS_ID"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]                = "1"

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from debate_engine_final2 import run_debate
from model_inference import load_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "debate_eval_results_Qwen_light_multirebuttal.csv")

load_dotenv()

hf_client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv("HF_TOKEN"),
)
logger.info("HF Inference client ready — using Meta-Llama-3-8B-Instruct")

df = pd.read_csv(os.path.join(BASE_DIR, "pubmedqa_train.csv"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_verdict_answer(result: dict) -> str:
    """Pull a normalised yes/no string out of whatever run_debate returns."""
    verdict = result.get("verdict") if isinstance(result, dict) else None
    if isinstance(verdict, dict) and verdict.get("answer") is not None:
        return str(verdict["answer"]).lower()

    direct_answer = result.get("verdict_answer") if isinstance(result, dict) else None
    if direct_answer is not None:
        return str(direct_answer).lower()

    return ""


def _make_entry(index: int, question: str, decision: str) -> dict:
    """Return a result-row skeleton with safe defaults."""
    return {
        "index"    : index,
        "question" : question,
        "expected" : decision.lower(),
        "verdict"  : None,
        "correct"  : False,
        "score"    : None,
        "avg_score": None,
        "reasoning": None,
    }


def _append_and_maybe_flush(
    entry      : dict,
    results_log: list,
    save_every : int,
    total      : int,
    error_count: int,
) -> list:
    """
    Append *entry* to *results_log* and flush to disk when the buffer
    reaches *save_every* rows.  Returns the (possibly cleared) buffer.
    Centralising the flush here means skipped rows and error rows are
    saved on exactly the same cadence as scored rows — no row is lost
    if the process dies mid-run.
    """
    results_log.append(entry)
    if len(results_log) >= save_every:
        save_results(results_log)
        logger.info("--- Progress: %d valid / %d errors ---", total, error_count)
        return []
    return results_log


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(results_log: list) -> None:
    """Append *results_log* rows to the CSV, writing the header only once."""
    if not results_log:
        logger.debug("save_results called with empty list — nothing written.")
        return

    results_df  = pd.DataFrame(results_log)
    file_exists = os.path.exists(LOG_PATH)
    results_df.to_csv(
        LOG_PATH,
        mode   = "a" if file_exists else "w",
        header = not file_exists,
        index  = False,
    )
    logger.info("Saved %d row(s) to %s", len(results_df), LOG_PATH)


def print_score_distribution() -> None:
    """Print a score-band histogram from the persisted CSV."""
    if not os.path.exists(LOG_PATH):
        logger.warning("No results file at %s — skipping distribution.", LOG_PATH)
        return

    full_df = pd.read_csv(LOG_PATH)
    scored  = full_df[full_df["score"].notna()].copy()
    if scored.empty:
        logger.info("No scored rows yet — skipping distribution.")
        return

    bins   = [0, 2, 4, 6, 8, 10]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]
    scored["score_band"] = pd.cut(
        scored["score"], bins=bins, labels=labels, include_lowest=True
    )
    print("\nScore Distribution:")
    print(scored["score_band"].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def run_judge(ground_truth: str, final_answer: str, question: str) -> tuple[float, str]:
    """
    Ask Llama-3-8B to score the debate verdict against the ground truth.
    Returns (score_0_to_10, one_sentence_reason).
    """
    response = hf_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict medical reasoning judge. "
                    "Always reply in the exact format requested. "
                    "Never add extra text outside the format."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question        : {question}\n"
                    f"Ground Truth    : {ground_truth}\n"
                    f"Debate Verdict  : {final_answer}\n\n"
                    "Score the debate verdict out of 10 based on:\n"
                    "- Factual alignment with ground truth   (5 points)\n"
                    "- Logical reasoning quality             (5 points)\n\n"
                    "Reply in this EXACT format and nothing else:\n"
                    "SCORE: <number between 0 and 10>\n"
                    "REASON: <one sentence explanation>"
                ),
            },
        ],
        max_tokens=80,
        temperature=0.1,
        seed=42,
    )

    decoded      = response.choices[0].message.content.strip()
    score_match  = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", decoded)
    reason_match = re.search(r"REASON:\s*(.+)", decoded, re.IGNORECASE)

    score     = float(score_match.group(1)) if score_match else 0.0
    score     = max(0.0, min(10.0, score))
    reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"

    return score, reasoning


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_debate_on_pubmedqa(
    start     : int = 0,
    no_rows   : int = 5,
    save_every: int = 2,
) -> tuple[float, float]:
    correct     = 0
    total       = 0
    error_count = 0
    score_sum   = 0.0
    results_log: list[dict] = []

    if save_every <= 0:
        logger.warning("Invalid --save-every=%d; defaulting to 1.", save_every)
        save_every = 1

    model_path = os.path.join(BASE_DIR, "argument_quality_model_4features.pth")
    try:
        load_models(model_path)
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        return 0.0, 0.0

    for index, row in df.iloc[start : start + no_rows].iterrows():
        question = row["question"]
        answer   = row["long_answer"]
        decision = row["final_decision"]

        entry = _make_entry(index, question, decision)

        try:
            result  = run_debate(topic=question, max_arg_iterations=5)
            verdict = _extract_verdict_answer(result)
            entry["verdict"] = verdict

            if verdict not in ("yes", "no"):
                logger.warning("[%4d] Skipping — verdict not yes/no: '%s'", index, verdict)
                entry["reasoning"] = "Skipped — verdict not yes/no"
                entry["avg_score"] = score_sum / total if total > 0 else 0.0
                results_log = _append_and_maybe_flush(
                    entry, results_log, save_every, total, error_count
                )
                continue

            score, reasoning = run_judge(
                ground_truth=answer,
                final_answer=verdict,
                question=question,
            )

            total     += 1
            score_sum += score
            avg_score  = score_sum / total
            is_correct = verdict == decision.lower()
            if is_correct:
                correct += 1

            entry.update({
                "correct"  : is_correct,
                "score"    : score,
                "avg_score": avg_score,
                "reasoning": reasoning,
            })

            status = "✅" if is_correct else "❌"
            logger.info(
                "%s [%4d] Score: %5.1f/10 | Avg: %5.2f/10 | Accuracy: %d/%d | "
                "Expected: %s | Verdict: %s | Reason: %s | Question: %s",
                status, index, score, avg_score, correct, total,
                decision.lower(), verdict, reasoning, question,
            )

        except Exception as e:
            error_count += 1
            logger.warning("[%4d] Error (excluded from score/accuracy): %s", index, e)
            entry.update({
                "verdict"  : "error",
                "avg_score": score_sum / total if total > 0 else 0.0,
                "reasoning": f"Error: {e}",
            })

        # Both the normal path and the error path converge here.
        results_log = _append_and_maybe_flush(
            entry, results_log, save_every, total, error_count
        )

    # Flush whatever is left in the buffer after the loop finishes.
    save_results(results_log)

    final_avg = score_sum / total if total > 0 else 0.0
    accuracy  = correct   / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Total Evaluated : {total}")
    print(f"  Errors Skipped  : {error_count}")
    print(f"  Correct Verdicts: {correct}/{total} = {accuracy:.2%}")
    print(f"  Final Avg Score : {final_avg:.2f}/10")
    print(f"  Judge backend   : Llama 3 8B (HF Inference API)")
    print(f"{'='*60}")

    print_score_distribution()
    return final_avg, accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run debate evaluation on PubMedQA.")
    parser.add_argument("--start",      type=int, default=0, help="Start row index")
    parser.add_argument("--no-rows",    type=int, default=5, help="Number of rows to evaluate")
    parser.add_argument("--save-every", type=int, default=2, help="Flush to CSV every N rows")
    args = parser.parse_args()

    final_avg, accuracy = run_debate_on_pubmedqa(
        start     = args.start,
        no_rows   = args.no_rows,
        save_every = args.save_every,
    )
    print(f"\nFinal Average Score: {final_avg:.2f}/10")
    print(f"Final Accuracy     : {accuracy:.2%}")