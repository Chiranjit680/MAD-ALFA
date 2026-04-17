import os

os.environ["CUDA_VISIBLE_DEVICES"]              = "5"
os.environ["CUDA_DEVICE_ORDER"]                 = "PCI_BUS_ID"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **kw: None

from debate_engine import run_debate
import pandas as pd
from dotenv import load_dotenv
from model_inference import load_models
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from google import genai
import torch
import re

load_dotenv()

free_vram  = torch.cuda.mem_get_info(0)[0] / 1024**3
total_vram = torch.cuda.mem_get_info(0)[1] / 1024**3
print(f"GPU       : {torch.cuda.get_device_name(0)}")
print(f"Free VRAM : {free_vram:.1f} / {total_vram:.1f} GB")

df = pd.read_csv("pubmedqa_train.csv")
model_path = "mixtral_local"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

def make_device_map(free_gb: float) -> dict | str:
    if free_gb >= 24:
        print("✅ Enough VRAM — loading fully on GPU")
        return {"": 0}
    else:
        print(f"⚠️  Only {free_gb:.1f}GB free — using CPU offload for overflow layers")
        return {
            "model.embed_tokens" : "cpu",
            "model.norm"         : 0,
            "lm_head"            : "cpu",
            "model.layers"       : 0,
        }

judge_model = None

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading judge model...")
try:
    judge_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=make_device_map(free_vram),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        offload_folder="./offload_cache",
        offload_state_dict=True,
    )
    judge_model.eval()
    print(f"✅ Local model loaded — VRAM: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
    USE_GEMINI = False

except Exception as load_err:
    print(f"❌ Local model failed: {load_err}")
    print("→ Falling back to Gemini API")
    judge_model = None
    USE_GEMINI  = True

# ✅ Gemini client — only initialised if needed
if USE_GEMINI:
    # reads GEMINI_API_KEY from .env automatically
    gemini_client = genai.Client(api_key="AIzaSyDzny21HlBRAMXMP4mJt6UXkdZ5MD5CMWo")
    print("✅ Gemini fallback active — using gemini-2.0-flash\n")


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

    # ─── Local Mixtral path ───────────────────────────────────────
    if not USE_GEMINI:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to("cuda:0")

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = outputs[0][inputs['input_ids'].shape[1]:]
        decoded   = tokenizer.decode(generated, skip_special_tokens=True).strip()
        torch.cuda.empty_cache()

    # ─── Gemini fallback path ─────────────────────────────────────
    else:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        decoded = response.text.strip()

    # ─── Parse output (same for both paths) ──────────────────────
    score_match  = re.search(r'SCORE:\s*([0-9]+(?:\.[0-9]+)?)', decoded)
    reason_match = re.search(r'REASON:\s*(.+)', decoded, re.IGNORECASE)

    score     = float(score_match.group(1))   if score_match  else 0.0
    score     = max(0.0, min(10.0, score))
    reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"

    return score, reasoning


def run_debate_on_pubmedqa():
    no_rows   = 2
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

            status  = "✅" if is_correct else "❌"
            backend = "gemini" if USE_GEMINI else "local"
            vram_str = (
                f"VRAM: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB"
                if not USE_GEMINI else "VRAM: N/A (Gemini)"
            )
            print(
                f"{status} [{index:>4}] "
                f"Score: {score:>5.1f}/10 | "
                f"Avg: {avg_score:>5.2f}/10 | "
                f"Accuracy: {correct}/{total} | "
                f"{vram_str} | "
                f"[{backend}] Reason: {reasoning}"
            )

            results_log.append({
                "index"    : index,
                "question" : question,
                "expected" : expected,
                "verdict"  : verdict,
                "correct"  : is_correct,
                "score"    : score,
                "avg_score": avg_score,
                "reasoning": reasoning,
                "backend"  : backend
            })

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            total += 1
            free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
            print(f"🔴 OOM row {index} — free VRAM: {free_gb:.1f}GB")
            results_log.append({
                "index"    : index,
                "question" : question,
                "expected" : decision.lower(),
                "verdict"  : "oom_error",
                "correct"  : False,
                "score"    : 0.0,
                "avg_score": score_sum / total if total > 0 else 0.0,
                "reasoning": "CUDA Out of Memory",
                "backend"  : "local"
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
                "reasoning": f"Error: {str(e)}",
                "backend"  : "unknown"
            })

    final_avg = score_sum / total if total > 0 else 0.0
    accuracy  = correct   / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Total Evaluated : {total}")
    print(f"  Correct Verdicts: {correct}/{total} = {accuracy:.2%}")
    print(f"  Final Avg Score : {final_avg:.2f}/10")
    if not USE_GEMINI:
        print(f"  Peak VRAM used  : {torch.cuda.max_memory_allocated(0)/1024**3:.1f} GB")
    print(f"  Judge backend   : {'Gemini API' if USE_GEMINI else 'Local Mixtral'}")
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
    os.makedirs("./offload_cache", exist_ok=True)
    final_avg, accuracy, log = run_debate_on_pubmedqa()