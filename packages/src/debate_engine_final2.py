"""
Full Debate Orchestrator Graph (LangGraph)
==========================================

Pipeline:
  start
    │
    ▼
  moderator_node            ← sets topic, extracts context, frames the debate
    │
    ▼
  debater1_arg_node         ← Debater 1 (PRO)  → debate_argument_generator subgraph
    │
    ▼
  score_pro_arg_node        ← Moderator scores PRO opening argument
    │
    ▼
  debater2_arg_node         ← Debater 2 (CON)  → debate_argument_generator subgraph
    │
    ▼
  score_con_arg_node        ← Moderator scores CON opening argument
    │
    ▼
  debater1_rebuttal_node    ← Debater 1 rebuts CON opening arg  → rebuttal subgraph
    │
    ▼
  score_pro_rebuttal_node   ← Moderator scores PRO rebuttal (round 1)
    │
    ▼
  debater2_rebuttal_node    ← Debater 2 rebuts PRO opening arg  → rebuttal subgraph
    │
    ▼
  score_con_rebuttal_node   ← Moderator scores CON rebuttal (round 1)
    │
    ▼
  debater1_rebuttal2_node   ← Debater 1 rebuts CON *rebuttal*   → rebuttal subgraph
    │
    ▼
  score_pro_rebuttal2_node  ← Moderator scores PRO rebuttal (round 2)
    │
    ▼
  debater2_rebuttal2_node   ← Debater 2 rebuts PRO *rebuttal*   → rebuttal subgraph
    │
    ▼
  score_con_rebuttal2_node  ← Moderator scores CON rebuttal (round 2)
    │
    ▼
  convergence_node          ← Moderator tallies scores + declares outcome
    │
    ▼
  verdict_node              ← Final YES/NO answer + elaborate explanation
    │
    ▼
  END
"""

from __future__ import annotations

import json
import os
import re
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

import numpy as np
import torch
from langgraph.graph import StateGraph, END
from transformers import pipeline as hf_pipeline
from model_inference import load_models

try:
    from .local_model_store import get_local_model_dir
except ImportError:
    from local_model_store import get_local_model_dir

try:
    from .get_judge_lm import llm_inference
except ImportError:
    from get_judge_lm import llm_inference

from debate_agent import DebateState, create_debate_argument_graph
from rebuttal_node import RebuttalState, create_rebuttal_graph

load_dotenv()

# =============================================================================
# Runtime model configuration
# =============================================================================
_HAS_CUDA = torch.cuda.is_available()
# Respect externally pinned CUDA visibility (e.g. CUDA_VISIBLE_DEVICES in
# debate_benchmarking_hf.py). After masking, selected GPU is always local cuda:0.
_DEVICE_ID = 0 if _HAS_CUDA else -1
_DTYPE = torch.float16 if _HAS_CUDA else torch.float32

_NLI_MODEL_ID   = "facebook/bart-large-mnli"
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_nli_pipeline   = None
_embed_pipeline = None

# Allowed values — used for validation after every LLM parse
_VALID_WINNERS  = {"PRO", "CON", "DRAW"}
_VALID_VERDICTS = {"YES", "NO", "INCONCLUSIVE"}


# =============================================================================
# Utilities
# =============================================================================

def _trunc(text: str, limit: int = 600) -> str:
    """Hard-truncate text to avoid blowing the context window."""
    return (text[:limit] + "…") if len(text) > limit else text


def _load_pipeline_with_fallback(task, model_path, device_id, dtype, **kwargs):
    """Load a HF pipeline; retry on CPU if GPU OOM."""
    try:
        return hf_pipeline(task, model=model_path, device=device_id,
                           torch_dtype=dtype, **kwargs)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device_id != -1:
            print(f"   ⚠️  OOM loading {model_path} on GPU — retrying on CPU.")
            torch.cuda.empty_cache()
            return hf_pipeline(task, model=model_path, device=-1,
                               torch_dtype=torch.float32, **kwargs)
        raise


# =============================================================================
# Lazy model initialisation
# =============================================================================

def initialize_runtime_models(force_reload: bool = False) -> None:
    global _nli_pipeline, _embed_pipeline

    if not force_reload and _nli_pipeline is not None and _embed_pipeline is not None:
        return

    print("\n🚀 Initializing debate engine runtime models...")
    print(f"   CUDA available : {_HAS_CUDA}")
    if _HAS_CUDA:
        print(f"   GPU device ID  : {_DEVICE_ID}")
        print(f"   GPU name       : {torch.cuda.get_device_name(_DEVICE_ID)}")

    if force_reload or _nli_pipeline is None:
        print(f"   Loading NLI    : {_NLI_MODEL_ID}")
        nli_path = get_local_model_dir(_NLI_MODEL_ID)
        _nli_pipeline = _load_pipeline_with_fallback(
            "zero-shot-classification", nli_path, _DEVICE_ID, _DTYPE)
        print("   ✅ NLI model ready")

    if force_reload or _embed_pipeline is None:
        print(f"   Loading Embed  : {_EMBED_MODEL_ID}")
        embed_path = get_local_model_dir(_EMBED_MODEL_ID)
        _embed_pipeline = _load_pipeline_with_fallback(
            "feature-extraction", embed_path, _DEVICE_ID, _DTYPE,
            return_tensor=False)
        print("   ✅ Embedding model ready")

    print("✅ All debate engine runtime models loaded.\n")


# =============================================================================
# Orchestrator State
# =============================================================================

class OrchestratorState(TypedDict):
    # inputs
    topic:              str
    max_arg_iterations: int

    # moderator
    framed_topic:    str
    context_summary: str

    # debater 1 (PRO) – argument
    d1_argument:           str
    d1_quality_score:      float
    d1_quality_level:      str
    d1_pubmed_query:       str
    d1_iterations_used:    int
    d1_retrieval_attempts: int
    d1_accepted:           bool
    d1_scores:             dict
    d1_evidence:           list

    # debater 2 (CON) – argument
    d2_argument:           str
    d2_quality_score:      float
    d2_quality_level:      str
    d2_pubmed_query:       str
    d2_iterations_used:    int
    d2_retrieval_attempts: int
    d2_accepted:           bool
    d2_scores:             dict
    d2_evidence:           list

    # debater 1 (PRO) – rebuttal round 1  (targets CON opening argument)
    d1_rebuttal:                str
    d1_rebuttal_logical_flaws:  list
    d1_rebuttal_counter_points: list
    d1_rebuttal_evidence:       dict

    # debater 2 (CON) – rebuttal round 1  (targets PRO opening argument)
    d2_rebuttal:                str
    d2_rebuttal_logical_flaws:  list
    d2_rebuttal_counter_points: list
    d2_rebuttal_evidence:       dict

    # debater 1 (PRO) – rebuttal round 2  (targets CON round-1 rebuttal)
    d1_rebuttal2:                str
    d1_rebuttal2_logical_flaws:  list
    d1_rebuttal2_counter_points: list
    d1_rebuttal2_evidence:       dict

    # debater 2 (CON) – rebuttal round 2  (targets PRO round-1 rebuttal)
    d2_rebuttal2:                str
    d2_rebuttal2_logical_flaws:  list
    d2_rebuttal2_counter_points: list
    d2_rebuttal2_evidence:       dict

    # moderator scores out of 10 — one per speech (6 speeches total)
    score_pro_arg:       float   # PRO opening argument
    score_con_arg:       float   # CON opening argument
    score_pro_rebuttal:  float   # PRO rebuttal round 1
    score_con_rebuttal:  float   # CON rebuttal round 1
    score_pro_rebuttal2: float   # PRO rebuttal round 2
    score_con_rebuttal2: float   # CON rebuttal round 2

    # convergence
    convergence_summary: str
    winner:              str

    # local-model scores
    pro_nli_score:      float
    con_nli_score:      float
    pro_semantic_score: float
    con_semantic_score: float

    # verdict
    verdict_answer:    str
    verdict_brief:     str
    verdict_elaborate: str

    # trace
    messages: Annotated[list, operator.add]


# =============================================================================
# LLM helper — local Mistral-7B
# =============================================================================

def _local_chat(messages: list[dict], max_new_tokens: int = 600) -> str:
    """
    Call mistral_inference with the conversation messages.

    mistral_inference is assumed to accept either:
      (a) a pre-formatted prompt string, OR
      (b) a list of {"role": ..., "content": ...} dicts.

    We try (b) first; if the function signature only accepts a string we
    fall back to (a) with minimal Mistral-instruct formatting.

    IMPORTANT: We deliberately do NOT double-wrap with <s>[INST]…[/INST]
    if mistral_inference already applies its own chat template internally.
    Inspect get_judge_lm.py if you still get empty responses — the template
    may need to be removed from one side.
    """
    import inspect
    sig = inspect.signature(llm_inference)
    params = list(sig.parameters.keys())

    first_param = params[0] if params else "prompt"
    if first_param in ("messages", "conversation"):
        raw = llm_inference(messages, max_new_tokens=max_new_tokens)
    else:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_prefix = (" ".join(system_parts) + "\n\n") if system_parts else ""
        turns = []
        for m in messages:
            if m["role"] == "user":
                turns.append(f"[INST] {m['content']} [/INST]")
            elif m["role"] == "assistant":
                turns.append(m["content"])
        prompt = "<s>" + system_prefix + " ".join(turns)
        raw = llm_inference(prompt, max_new_tokens=max_new_tokens)

    print(f"   📡 llm_inference called (max_new_tokens={max_new_tokens})")
    result = (raw or "").strip()
    if not result:
        print("   ⚠️  llm_inference returned an empty string.")
    else:
        print(f"   ✅ Response received ({len(result)} chars).")
    return result


def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.

    Strategy:
      1. Strip markdown fences (```json ... ```)
      2. Try direct json.loads
      3. Extract first {...} block and retry
      4. Raise with the original text included so callers can log it
    """
    if not raw or not raw.strip():
        raise ValueError(f"LLM returned empty text — cannot parse JSON.\nRaw: {raw!r}")

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    repaired = cleaned
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"Could not parse JSON from LLM output.\n"
        f"--- Raw output (first 500 chars) ---\n{raw[:500]}"
    )


def _llm_call_with_retry(
    messages: list[dict],
    max_new_tokens: int,
    required_keys: list[str],
    node_name: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> tuple[dict, str]:
    """
    Call the LLM and parse JSON, retrying up to max_retries times.
    On each retry appends the failed response + correction nudge so the
    model can self-correct.
    """
    import time

    last_exc: Exception = RuntimeError("No attempts made.")
    last_raw: str = ""
    conversation = list(messages)

    for attempt in range(1, max_retries + 1):
        try:
            raw       = _local_chat(conversation, max_new_tokens=max_new_tokens)
            last_raw  = raw
            parsed    = _extract_json(raw)

            missing = [k for k in required_keys if not parsed.get(k)]
            if missing:
                raise ValueError(
                    f"Parsed JSON missing required keys: {missing}\n"
                    f"Got keys: {list(parsed.keys())}"
                )

            if attempt > 1:
                print(f"   ✅ [{node_name}] Succeeded on attempt {attempt}/{max_retries}.")
            return parsed, raw

        except Exception as exc:
            last_exc = exc
            last_raw = locals().get("raw", "")
            print(f"   ⚠  [{node_name}] Attempt {attempt}/{max_retries} failed: {exc}")
            if last_raw:
                print(f"   ⚠  Raw output (first 300 chars): {last_raw[:300]!r}")

            if attempt < max_retries:
                wait = backoff_seconds * attempt
                print(f"   🔄 Retrying in {wait:.1f}s …")
                time.sleep(wait)

                conversation = list(messages) + [
                    {"role": "assistant", "content": last_raw or "(empty response)"},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as valid JSON. "
                            f"Error: {exc}\n\n"
                            "Please respond ONLY with a valid JSON object — "
                            "no explanation, no markdown fences, no extra text."
                        ),
                    },
                ]

    raise last_exc


# =============================================================================
# Fallback helpers
# =============================================================================

def _fallback_moderator(topic: str) -> dict:
    return {
        "framed_topic": (
            f"Should the claim about '{topic}' be accepted "
            "based on the available evidence?"
        ),
        "context_summary": (
            "This debate examines the strength of the current evidence "
            "supporting the topic, with attention to both direct findings "
            "and alternative explanations."
        ),
    }


def _fallback_convergence_summary(scores: dict) -> dict:
    """Derive winner from automated scores when the LLM fails."""
    pro_total = (
        float(scores.get("pro_nli_score",      0.0)) +
        float(scores.get("pro_semantic_score", 0.0))
    )
    con_total = (
        float(scores.get("con_nli_score",      0.0)) +
        float(scores.get("con_semantic_score", 0.0))
    )

    if pro_total > con_total + 0.05:
        winner = "PRO"
    elif con_total > pro_total + 0.05:
        winner = "CON"
    else:
        winner = "DRAW"

    summary = (
        "The model returned non-JSON text during convergence, so the result "
        "was derived from the automated local scores instead. "
        f"PRO combined score: {pro_total:.3f}. "
        f"CON combined score: {con_total:.3f}."
    )
    justification = (
        f"Score comparison favoured {winner} (PRO={pro_total:.3f}, CON={con_total:.3f})."
        if winner != "DRAW"
        else "Automated scores were closely balanced — treated as a draw."
    )
    return {
        "convergence_summary":  summary,
        "winner":               winner,
        "winner_justification": justification,
    }


def _fallback_verdict(state: OrchestratorState) -> dict:
    winner = str(state.get("winner", "DRAW")).upper()
    answer = {"PRO": "YES", "CON": "NO"}.get(winner, "INCONCLUSIVE")
    brief = (
        "The available debate evidence does not support a single "
        "high-confidence binary conclusion."
        if answer == "INCONCLUSIVE"
        else f"Based on the debate synthesis, the best-supported answer is {answer}."
    )
    elaborate = (
        "The final verdict was produced via fallback logic because the LLM "
        "response could not be parsed as valid JSON in the verdict node. "
        "The fallback uses the convergence winner and local scoring context. "
        "Review the raw output logs above and consider reducing prompt length "
        "or max_new_tokens if you need a fully model-authored verdict."
    )
    return {
        "verdict_answer":    answer,
        "verdict_brief":     brief,
        "verdict_elaborate": elaborate,
    }


# =============================================================================
# Local-model scoring helpers
# =============================================================================

def _nli_score(premise: str, hypothesis: str) -> float:
    if _nli_pipeline is None:
        initialize_runtime_models()
    try:
        result = _nli_pipeline(
            premise[:1024],
            candidate_labels=[hypothesis[:200]],
            multi_label=False,
        )
        return float(result["scores"][0])
    except Exception as exc:
        print(f"   ⚠️  NLI scoring error: {exc}")
        return 0.0


def _mean_pooling(token_embeddings) -> np.ndarray:
    arr = np.array(token_embeddings)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.mean(axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def _semantic_score(text_a: str, text_b: str) -> float:
    if _embed_pipeline is None:
        initialize_runtime_models()
    try:
        emb_a = _mean_pooling(_embed_pipeline(text_a[:512]))
        emb_b = _mean_pooling(_embed_pipeline(text_b[:512]))
        return _cosine_similarity(emb_a, emb_b)
    except Exception as exc:
        print(f"   ⚠️  Semantic scoring error: {exc}")
        return 0.0


def _score_argument_pair(
    framed_topic: str,
    pro_argument: str,
    con_argument: str,
) -> dict:
    print("\n   🔬 Scoring arguments with local models …")
    pro_nli = _nli_score(pro_argument, framed_topic)
    con_nli = _nli_score(con_argument, framed_topic)
    print(f"   NLI entailment — PRO: {pro_nli:.3f}  CON: {con_nli:.3f}")

    pro_sem = _semantic_score(pro_argument, framed_topic)
    con_sem = _semantic_score(con_argument, framed_topic)
    print(f"   Semantic sim   — PRO: {pro_sem:.3f}  CON: {con_sem:.3f}")

    return {
        "pro_nli_score":      pro_nli,
        "con_nli_score":      con_nli,
        "pro_semantic_score": pro_sem,
        "con_semantic_score": con_sem,
    }


# =============================================================================
# Shared rebuttal helper — invoke the rebuttal subgraph and package results
# =============================================================================

def _run_rebuttal(
    target_argument: str,
    framed_topic: str,
    role_label: str,    # "PRO" or "CON"
    round_label: str,   # "round 1" or "round 2"
) -> dict:
    """
    Invoke create_rebuttal_graph() against *target_argument* and return a
    plain dict with keys: rebuttal, logical_flaws, counter_points, evidence_counts.
    """
    initial: RebuttalState = {
        "original_argument": target_argument,
        "topic":             framed_topic,
        "logical_flaws":     [],
        "counter_points":    [],
        "pubmed_evidence":   {},
        "rebuttal":          "",
        "messages":          [],
    }
    final = create_rebuttal_graph().invoke(initial)
    evidence_counts = {
        cp: len(abstracts)
        for cp, abstracts in final["pubmed_evidence"].items()
    }

    suffix = "…" if len(final["rebuttal"]) > 200 else ""
    print(f"\n   ✅ {role_label} Rebuttal ({round_label}):\n"
          f"   {final['rebuttal'][:200]}{suffix}")
    print(f"   🔍 Logical flaws   : {len(final['logical_flaws'])}")
    print(f"   💡 Counter-points  : {len(final['counter_points'])}")
    print(f"   📚 PubMed abstracts: {sum(evidence_counts.values())}\n")

    return {
        "rebuttal":        final["rebuttal"],
        "logical_flaws":   final["logical_flaws"],
        "counter_points":  final["counter_points"],
        "evidence_counts": evidence_counts,
    }


# =============================================================================
# Scoring helper — moderator scores a single argument or rebuttal out of 10
# =============================================================================

def _moderator_score(role: str, text: str, topic: str, context: str = "") -> float:
    """
    Ask the moderator to score a piece of debate text out of 10.
    Falls back to 5.0 if no numeric value is found in the response.

    Parameters
    ----------
    role    : human-readable label, e.g. "PRO argument" or "CON rebuttal round 2"
    text    : the argument or rebuttal text to score
    topic   : the framed debate motion
    context : optional — the opposing speech that this text is responding to
    """
    prompt = (
        f"You are a strict debate judge scoring a {role} on the motion:\n"
        f"\"{topic}\"\n\n"
    )
    if context:
        prompt += f"For reference, the opposing speech was:\n{_trunc(context, 300)}\n\n"
    prompt += (
        f"{role.upper()}:\n{_trunc(text, 500)}\n\n"
        f"Score this {role} out of 10 based on:\n"
        "  • Strength and relevance of evidence\n"
        "  • Logical coherence\n"
        "  • Directly addressing the motion\n"
        "  • Clarity and persuasiveness\n\n"
        "Reply with ONLY a single number between 0 and 10 (decimals allowed). "
        "No explanation, no text — just the number."
    )

    raw = _local_chat([{"role": "user", "content": prompt}], max_new_tokens=10)

    match = re.search(r"\b(\d{1,2}(?:\.\d{1,2})?)\b", raw or "")
    if match:
        score = float(match.group(1))
        score = max(0.0, min(10.0, score))
        print(f"   🎯 Moderator score for {role}: {score:.1f}/10")
        return score

    print(f"   ⚠  Could not parse score from: {raw!r} — defaulting to 5.0")
    return 5.0


# =============================================================================
# Node 1 – Moderator
# =============================================================================

def moderator_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR NODE – Framing the debate")
    print(f"{'='*70}")

    prompt = (
        "You are a professional debate moderator.\n\n"
        "Given the following debate topic, your tasks are:\n"
        "1. Frame it as a clear, neutral debate motion (one sentence).\n"
        "2. Write a 2-3 sentence background summary that contextualises "
        "the topic without taking sides.\n\n"
        f"Topic: {state['topic']}\n\n"
        "You MUST respond with ONLY a valid JSON object — no explanation, "
        "no markdown fences, no extra text before or after:\n"
        '{"framed_topic": "...", "context_summary": "..."}'
    )

    framed_topic    = ""
    context_summary = ""
    raw             = ""
    try:
        raw             = _local_chat([{"role": "user", "content": prompt}], max_new_tokens=300)
        parsed          = _extract_json(raw)
        framed_topic    = parsed["framed_topic"]
        context_summary = parsed["context_summary"]
    except Exception as exc:
        print(f"   ⚠  Moderator parse failed: {exc}")
        print(f"   ⚠  Raw output: {raw[:400]!r}")
        fallback        = _fallback_moderator(state["topic"])
        framed_topic    = fallback["framed_topic"]
        context_summary = fallback["context_summary"]

    print(f"\n   📌 Framed Topic    : {framed_topic}")
    print(f"   📖 Context Summary : {context_summary}")

    return {
        "framed_topic":    framed_topic,
        "context_summary": context_summary,
        "messages": [{"role": "moderator",
                      "content": f"Debate framed — Motion: {framed_topic}"}],
    }


# =============================================================================
# Node 2 – Debater 1 (PRO) – Argument
# =============================================================================

def debater1_arg_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔵 DEBATER 1 (PRO) – Generating opening argument")
    print(f"{'='*70}")

    initial: DebateState = {
        "topic":              state["framed_topic"],
        "stance":             "PRO",
        "pubmed_query":       "",
        "argument":           "",
        "iteration":          1,
        "max_iterations":     state.get("max_arg_iterations", 5),
        "pubmed_abstracts":   [],
        "reranked_evidence":  [],
        "nli_score":          0.0,
        "semantic_score":     0.0,
        "logprob_score":      0.0,
        "confidence_score":   0.0,
        "quality_score":      0.0,
        "quality_level":      "",
        "accepted":           False,
        "feedback":           "",
        "prev_arg":           "",
        "opposition_arg":     "",
        "messages":           [],
        "retrieval_feedback": "",
        "retrieval_attempts": 0,
        "failure_reason":     "",
    }

    final  = create_debate_argument_graph().invoke(initial)
    suffix = "…" if len(final["argument"]) > 200 else ""
    print(f"\n   ✅ PRO Argument (quality={final['quality_score']:.3f}):\n"
          f"   {final['argument'][:200]}{suffix}")
    print(f"   📚 Evidence pieces: {len(final['reranked_evidence'])}\n")

    return {
        "d1_argument":           final["argument"],
        "d1_quality_score":      final["quality_score"],
        "d1_quality_level":      final["quality_level"],
        "d1_pubmed_query":       final["pubmed_query"],
        "d1_iterations_used":    final["iteration"],
        "d1_retrieval_attempts": final["retrieval_attempts"],
        "d1_accepted":           final["accepted"],
        "d1_scores": {
            "nli":        final["nli_score"],
            "semantic":   final["semantic_score"],
            "confidence": final["confidence_score"],
            "overall":    final["quality_score"],
        },
        "d1_evidence": final["reranked_evidence"],
        "messages": [{"role": "debater1_arg", "content": final["argument"]}],
    }


# =============================================================================
# Node 3 – Moderator scores PRO opening argument
# =============================================================================

def score_pro_arg_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – PRO Opening Argument")
    print(f"{'='*70}")
    score = _moderator_score(
        role  = "PRO argument",
        text  = state["d1_argument"],
        topic = state["framed_topic"],
    )
    return {
        "score_pro_arg": score,
        "messages": [{"role": "moderator_score",
                      "content": f"PRO argument score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 4 – Debater 2 (CON) – Argument
# =============================================================================

def debater2_arg_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔴 DEBATER 2 (CON) – Generating opening argument")
    print(f"{'='*70}")

    initial: DebateState = {
        "topic":              state["framed_topic"],
        "stance":             "CON",
        "pubmed_query":       "",
        "argument":           "",
        "iteration":          1,
        "max_iterations":     state.get("max_arg_iterations", 5),
        "pubmed_abstracts":   [],
        "reranked_evidence":  [],
        "nli_score":          0.0,
        "semantic_score":     0.0,
        "logprob_score":      0.0,
        "confidence_score":   0.0,
        "quality_score":      0.0,
        "quality_level":      "",
        "accepted":           False,
        "feedback":           "",
        "prev_arg":           "",
        "opposition_arg":     state["d1_argument"],   # second-speaker advantage
        "messages":           [],
        "retrieval_feedback": "",
        "retrieval_attempts": 0,
        "failure_reason":     "",
    }

    final  = create_debate_argument_graph().invoke(initial)
    suffix = "…" if len(final["argument"]) > 200 else ""
    print(f"\n   ✅ CON Argument (quality={final['quality_score']:.3f}):\n"
          f"   {final['argument'][:200]}{suffix}")
    print(f"   📚 Evidence pieces: {len(final['reranked_evidence'])}\n")

    return {
        "d2_argument":           final["argument"],
        "d2_quality_score":      final["quality_score"],
        "d2_quality_level":      final["quality_level"],
        "d2_pubmed_query":       final["pubmed_query"],
        "d2_iterations_used":    final["iteration"],
        "d2_retrieval_attempts": final["retrieval_attempts"],
        "d2_accepted":           final["accepted"],
        "d2_scores": {
            "nli":        final["nli_score"],
            "semantic":   final["semantic_score"],
            "confidence": final["confidence_score"],
            "overall":    final["quality_score"],
        },
        "d2_evidence": final["reranked_evidence"],
        "messages": [{"role": "debater2_arg", "content": final["argument"]}],
    }


# =============================================================================
# Node 5 – Moderator scores CON opening argument
# =============================================================================

def score_con_arg_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – CON Opening Argument")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "CON argument",
        text    = state["d2_argument"],
        topic   = state["framed_topic"],
        context = state["d1_argument"],
    )
    return {
        "score_con_arg": score,
        "messages": [{"role": "moderator_score",
                      "content": f"CON argument score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 6 – Debater 1 (PRO) – Rebuttal round 1  (targets CON opening arg)
# =============================================================================

def debater1_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔵 DEBATER 1 (PRO) – Rebuttal round 1 (targeting CON opening argument)")
    print(f"{'='*70}")

    r = _run_rebuttal(
        target_argument = state["d2_argument"],
        framed_topic    = state["framed_topic"],
        role_label      = "PRO",
        round_label     = "round 1",
    )
    return {
        "d1_rebuttal":                r["rebuttal"],
        "d1_rebuttal_logical_flaws":  r["logical_flaws"],
        "d1_rebuttal_counter_points": r["counter_points"],
        "d1_rebuttal_evidence":       r["evidence_counts"],
        "messages": [{"role": "debater1_rebuttal", "content": r["rebuttal"]}],
    }


# =============================================================================
# Node 7 – Moderator scores PRO rebuttal round 1
# =============================================================================

def score_pro_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – PRO Rebuttal (round 1)")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "PRO rebuttal round 1",
        text    = state["d1_rebuttal"],
        topic   = state["framed_topic"],
        context = state["d2_argument"],
    )
    return {
        "score_pro_rebuttal": score,
        "messages": [{"role": "moderator_score",
                      "content": f"PRO rebuttal (round 1) score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 8 – Debater 2 (CON) – Rebuttal round 1  (targets PRO opening arg)
# =============================================================================

def debater2_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔴 DEBATER 2 (CON) – Rebuttal round 1 (targeting PRO opening argument)")
    print(f"{'='*70}")

    r = _run_rebuttal(
        target_argument = state["d1_argument"],
        framed_topic    = state["framed_topic"],
        role_label      = "CON",
        round_label     = "round 1",
    )
    return {
        "d2_rebuttal":                r["rebuttal"],
        "d2_rebuttal_logical_flaws":  r["logical_flaws"],
        "d2_rebuttal_counter_points": r["counter_points"],
        "d2_rebuttal_evidence":       r["evidence_counts"],
        "messages": [{"role": "debater2_rebuttal", "content": r["rebuttal"]}],
    }


# =============================================================================
# Node 9 – Moderator scores CON rebuttal round 1
# =============================================================================

def score_con_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – CON Rebuttal (round 1)")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "CON rebuttal round 1",
        text    = state["d2_rebuttal"],
        topic   = state["framed_topic"],
        context = state["d1_argument"],
    )
    return {
        "score_con_rebuttal": score,
        "messages": [{"role": "moderator_score",
                      "content": f"CON rebuttal (round 1) score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 10 – Debater 1 (PRO) – Rebuttal round 2  (targets CON round-1 rebuttal)
# =============================================================================

def debater1_rebuttal2_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔵 DEBATER 1 (PRO) – Rebuttal round 2 (targeting CON round-1 rebuttal)")
    print(f"{'='*70}")

    r = _run_rebuttal(
        target_argument = state["d2_rebuttal"],      # ← CON's round-1 rebuttal
        framed_topic    = state["framed_topic"],
        role_label      = "PRO",
        round_label     = "round 2",
    )
    return {
        "d1_rebuttal2":                r["rebuttal"],
        "d1_rebuttal2_logical_flaws":  r["logical_flaws"],
        "d1_rebuttal2_counter_points": r["counter_points"],
        "d1_rebuttal2_evidence":       r["evidence_counts"],
        "messages": [{"role": "debater1_rebuttal2", "content": r["rebuttal"]}],
    }


# =============================================================================
# Node 11 – Moderator scores PRO rebuttal round 2
# =============================================================================

def score_pro_rebuttal2_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – PRO Rebuttal (round 2)")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "PRO rebuttal round 2",
        text    = state["d1_rebuttal2"],
        topic   = state["framed_topic"],
        context = state["d2_rebuttal"],    # the speech it was responding to
    )
    return {
        "score_pro_rebuttal2": score,
        "messages": [{"role": "moderator_score",
                      "content": f"PRO rebuttal (round 2) score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 12 – Debater 2 (CON) – Rebuttal round 2  (targets PRO round-1 rebuttal)
# =============================================================================

def debater2_rebuttal2_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔴 DEBATER 2 (CON) – Rebuttal round 2 (targeting PRO round-1 rebuttal)")
    print(f"{'='*70}")

    r = _run_rebuttal(
        target_argument = state["d1_rebuttal"],      # ← PRO's round-1 rebuttal
        framed_topic    = state["framed_topic"],
        role_label      = "CON",
        round_label     = "round 2",
    )
    return {
        "d2_rebuttal2":                r["rebuttal"],
        "d2_rebuttal2_logical_flaws":  r["logical_flaws"],
        "d2_rebuttal2_counter_points": r["counter_points"],
        "d2_rebuttal2_evidence":       r["evidence_counts"],
        "messages": [{"role": "debater2_rebuttal2", "content": r["rebuttal"]}],
    }


# =============================================================================
# Node 13 – Moderator scores CON rebuttal round 2
# =============================================================================

def score_con_rebuttal2_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – CON Rebuttal (round 2)")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "CON rebuttal round 2",
        text    = state["d2_rebuttal2"],
        topic   = state["framed_topic"],
        context = state["d1_rebuttal"],    # the speech it was responding to
    )
    return {
        "score_con_rebuttal2": score,
        "messages": [{"role": "moderator_score",
                      "content": f"CON rebuttal (round 2) score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 14 – Convergence  (winner decided by cumulative moderator scores)
# =============================================================================

def convergence_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("⚖️  CONVERGENCE NODE – Tallying scores and generating summary")
    print(f"{'='*70}")

    # ── Tally all six moderator scores ────────────────────────────────────────
    pro_total = (
        state["score_pro_arg"]
        + state["score_pro_rebuttal"]
        + state["score_pro_rebuttal2"]
    )
    con_total = (
        state["score_con_arg"]
        + state["score_con_rebuttal"]
        + state["score_con_rebuttal2"]
    )

    if pro_total > con_total:
        winner = "PRO"
    elif con_total > pro_total:
        winner = "CON"
    else:
        winner = "DRAW"

    print(f"\n   📊 Scorecard:")
    print(f"      PRO argument      : {state['score_pro_arg']:.1f}/10")
    print(f"      CON argument      : {state['score_con_arg']:.1f}/10")
    print(f"      PRO rebuttal R1   : {state['score_pro_rebuttal']:.1f}/10")
    print(f"      CON rebuttal R1   : {state['score_con_rebuttal']:.1f}/10")
    print(f"      PRO rebuttal R2   : {state['score_pro_rebuttal2']:.1f}/10")
    print(f"      CON rebuttal R2   : {state['score_con_rebuttal2']:.1f}/10")
    print(f"      ─────────────────────────────")
    print(f"      PRO total         : {pro_total:.1f}/30")
    print(f"      CON total         : {con_total:.1f}/30")
    print(f"   🏆 Winner            : {winner}")

    # ── Run local NLI + semantic scoring (kept for verdict node context) ──────
    scores = _score_argument_pair(
        framed_topic = state["framed_topic"],
        pro_argument = state["d1_argument"],
        con_argument = state["d2_argument"],
    )

    # ── Generate plain-text closing summary ───────────────────────────────────
    summary_prompt = (
        "You are a professional debate moderator writing a closing summary.\n\n"
        f"Motion: {state['framed_topic']}\n\n"
        f"PRO argument (scored {state['score_pro_arg']:.1f}/10):\n"
        f"{_trunc(state['d1_argument'], 350)}\n\n"
        f"CON argument (scored {state['score_con_arg']:.1f}/10):\n"
        f"{_trunc(state['d2_argument'], 350)}\n\n"
        f"PRO rebuttal round 1 (scored {state['score_pro_rebuttal']:.1f}/10):\n"
        f"{_trunc(state['d1_rebuttal'], 250)}\n\n"
        f"CON rebuttal round 1 (scored {state['score_con_rebuttal']:.1f}/10):\n"
        f"{_trunc(state['d2_rebuttal'], 250)}\n\n"
        f"PRO rebuttal round 2 (scored {state['score_pro_rebuttal2']:.1f}/10):\n"
        f"{_trunc(state['d1_rebuttal2'], 250)}\n\n"
        f"CON rebuttal round 2 (scored {state['score_con_rebuttal2']:.1f}/10):\n"
        f"{_trunc(state['d2_rebuttal2'], 250)}\n\n"
        f"Final scores — PRO: {pro_total:.1f}/30  |  CON: {con_total:.1f}/30\n"
        f"Winner: {winner}\n\n"
        "Write a 3-5 sentence closing summary of the debate highlighting the "
        "key clash points across both rebuttal rounds and why the scores fell "
        "the way they did. Plain prose only — no JSON, no bullet points."
    )

    raw_summary = _local_chat(
        [{"role": "user", "content": summary_prompt}], max_new_tokens=350
    )
    conv_summary = raw_summary.strip() if raw_summary.strip() else (
        f"The debate on '{state['framed_topic']}' concluded after two rebuttal rounds "
        f"with PRO scoring {pro_total:.1f}/30 and CON scoring {con_total:.1f}/30. "
        f"{winner} is declared the winner."
    )

    display_summary = (
        f"{conv_summary}\n\n"
        f"📊 Scores — PRO: {pro_total:.1f}/30  |  CON: {con_total:.1f}/30\n"
        f"🏆 Winner: {winner}"
    )

    print(f"\n   📋 Summary: {conv_summary[:200]}")

    return {
        "convergence_summary": conv_summary,
        "winner":              winner,
        "pro_nli_score":       scores["pro_nli_score"],
        "con_nli_score":       scores["con_nli_score"],
        "pro_semantic_score":  scores["pro_semantic_score"],
        "con_semantic_score":  scores["con_semantic_score"],
        "messages": [{"role": "convergence", "content": display_summary}],
    }


# =============================================================================
# Node 15 – Verdict
# =============================================================================

def verdict_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🏛️  VERDICT NODE – Generating final YES/NO answer")
    print(f"{'='*70}")

    def _fmt_evidence(evidence_list: list, max_items: int = 3) -> str:
        lines = []
        for i, ev in enumerate(evidence_list[:max_items], 1):
            if isinstance(ev, dict):
                title   = ev.get("title", "")
                snippet = _trunc(ev.get("abstract", ev.get("text", "")), 150)
                lines.append(f"  {i}. [{title}] {snippet}")
            else:
                lines.append(f"  {i}. {_trunc(str(ev), 150)}")
        return "\n".join(lines) or "  (none)"

    def _fmt_rebuttal_ev(ev_dict: dict) -> str:
        if not ev_dict:
            return "  (none)"
        return "\n".join(
            f"  • \"{cp}\": {cnt} abstract(s)"
            for cp, cnt in ev_dict.items()
        )

    pro_ev      = _fmt_evidence(state.get("d1_evidence", []))
    con_ev      = _fmt_evidence(state.get("d2_evidence", []))
    pro_reb_ev  = _fmt_rebuttal_ev(state.get("d1_rebuttal_evidence",  {}))
    con_reb_ev  = _fmt_rebuttal_ev(state.get("d2_rebuttal_evidence",  {}))
    pro_reb2_ev = _fmt_rebuttal_ev(state.get("d1_rebuttal2_evidence", {}))
    con_reb2_ev = _fmt_rebuttal_ev(state.get("d2_rebuttal2_evidence", {}))

    pro_nli = state.get("pro_nli_score", 0.0)
    con_nli = state.get("con_nli_score", 0.0)
    pro_sem = state.get("pro_semantic_score", 0.0)
    con_sem = state.get("con_semantic_score", 0.0)

    pro_total = (
        state["score_pro_arg"]
        + state["score_pro_rebuttal"]
        + state["score_pro_rebuttal2"]
    )
    con_total = (
        state["score_con_arg"]
        + state["score_con_rebuttal"]
        + state["score_con_rebuttal2"]
    )

    prompt = (
        "You are an expert biomedical analyst who has observed a structured debate.\n"
        "Deliver a definitive, evidence-grounded verdict on the original question.\n\n"
        f"ORIGINAL QUESTION: {state['topic']}\n"
        f"FRAMED MOTION: {state['framed_topic']}\n"
        f"BACKGROUND: {_trunc(state['context_summary'], 300)}\n\n"
        f"PRO ARGUMENT (quality {state['d1_quality_score']:.3f}, "
        f"moderator score {state['score_pro_arg']:.1f}/10):\n"
        f"{_trunc(state['d1_argument'], 350)}\n"
        f"PRO PubMed evidence:\n{pro_ev}\n\n"
        f"CON ARGUMENT (quality {state['d2_quality_score']:.3f}, "
        f"moderator score {state['score_con_arg']:.1f}/10):\n"
        f"{_trunc(state['d2_argument'], 350)}\n"
        f"CON PubMed evidence:\n{con_ev}\n\n"
        f"PRO REBUTTAL ROUND 1 (moderator score {state['score_pro_rebuttal']:.1f}/10):\n"
        f"{_trunc(state['d1_rebuttal'], 250)}\n"
        f"Evidence:\n{pro_reb_ev}\n\n"
        f"CON REBUTTAL ROUND 1 (moderator score {state['score_con_rebuttal']:.1f}/10):\n"
        f"{_trunc(state['d2_rebuttal'], 250)}\n"
        f"Evidence:\n{con_reb_ev}\n\n"
        f"PRO REBUTTAL ROUND 2 (moderator score {state['score_pro_rebuttal2']:.1f}/10):\n"
        f"{_trunc(state['d1_rebuttal2'], 250)}\n"
        f"Evidence:\n{pro_reb2_ev}\n\n"
        f"CON REBUTTAL ROUND 2 (moderator score {state['score_con_rebuttal2']:.1f}/10):\n"
        f"{_trunc(state['d2_rebuttal2'], 250)}\n"
        f"Evidence:\n{con_reb2_ev}\n\n"
        f"LOCAL SCORES — PRO NLI: {pro_nli:.3f}, PRO Semantic: {pro_sem:.3f}, "
        f"CON NLI: {con_nli:.3f}, CON Semantic: {con_sem:.3f}\n\n"
        f"MODERATOR TOTALS — PRO: {pro_total:.1f}/30  |  CON: {con_total:.1f}/30\n"
        f"CONVERGENCE: {_trunc(state['convergence_summary'], 300)}\n"
        f"DEBATE WINNER: {state['winner']}\n\n"
        "Based on ALL of the above, answer the original question.\n"
        "You MUST respond with ONLY a valid JSON object — no markdown, no extra text:\n"
        '{"verdict_answer": "YES or NO or INCONCLUSIVE", '
        '"verdict_brief": "one direct sentence", '
        '"verdict_elaborate": "150-250 word explanation"}'
    )

    verdict_answer    = "INCONCLUSIVE"
    verdict_brief     = ""
    verdict_elaborate = ""
    raw               = ""
    try:
        raw               = _local_chat([{"role": "user", "content": prompt}], max_new_tokens=600)
        parsed            = _extract_json(raw)
        answer_raw        = parsed.get("verdict_answer", "INCONCLUSIVE").strip().upper()
        verdict_answer    = answer_raw if answer_raw in _VALID_VERDICTS else "INCONCLUSIVE"
        verdict_brief     = parsed["verdict_brief"]
        verdict_elaborate = parsed["verdict_elaborate"]
    except Exception as exc:
        print(f"   ⚠  Verdict parse failed: {exc}")
        print(f"   ⚠  Raw output: {raw[:400]!r}")
        fallback          = _fallback_verdict(state)
        verdict_answer    = fallback["verdict_answer"]
        verdict_brief     = fallback["verdict_brief"]
        verdict_elaborate = fallback["verdict_elaborate"]

    badge = {"YES": "✅ YES", "NO": "❌ NO", "INCONCLUSIVE": "❓ INCONCLUSIVE"}.get(
        verdict_answer, verdict_answer
    )
    print(f"\n   🏛️  Verdict        : {badge}")
    print(f"   📝 Brief          : {verdict_brief}")
    print(f"\n   📖 Elaborate:\n")
    for para in verdict_elaborate.split("\n"):
        if para.strip():
            print(f"   {para.strip()}")

    return {
        "verdict_answer":    verdict_answer,
        "verdict_brief":     verdict_brief,
        "verdict_elaborate": verdict_elaborate,
        "messages": [{
            "role":    "verdict",
            "content": f"Verdict: {verdict_answer}\n\n{verdict_brief}\n\n{verdict_elaborate}",
        }],
    }


# =============================================================================
# Graph Assembly
# =============================================================================

def create_orchestrator_graph():
    workflow = StateGraph(OrchestratorState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    workflow.add_node("moderator",            moderator_node)
    workflow.add_node("debater1_arg",         debater1_arg_node)
    workflow.add_node("score_pro_arg",        score_pro_arg_node)
    workflow.add_node("debater2_arg",         debater2_arg_node)
    workflow.add_node("score_con_arg",        score_con_arg_node)
    workflow.add_node("debater1_rebuttal",    debater1_rebuttal_node)
    workflow.add_node("score_pro_rebuttal",   score_pro_rebuttal_node)
    workflow.add_node("debater2_rebuttal",    debater2_rebuttal_node)
    workflow.add_node("score_con_rebuttal",   score_con_rebuttal_node)
    workflow.add_node("debater1_rebuttal2",   debater1_rebuttal2_node)   # NEW
    workflow.add_node("score_pro_rebuttal2",  score_pro_rebuttal2_node)  # NEW
    workflow.add_node("debater2_rebuttal2",   debater2_rebuttal2_node)   # NEW
    workflow.add_node("score_con_rebuttal2",  score_con_rebuttal2_node)  # NEW
    workflow.add_node("convergence",          convergence_node)
    workflow.add_node("verdict",              verdict_node)

    # ── Wire edges ─────────────────────────────────────────────────────────────
    workflow.add_edge("moderator",            "debater1_arg")
    workflow.add_edge("debater1_arg",         "score_pro_arg")
    workflow.add_edge("score_pro_arg",        "debater2_arg")
    workflow.add_edge("debater2_arg",         "score_con_arg")
    workflow.add_edge("score_con_arg",        "debater1_rebuttal")
    workflow.add_edge("debater1_rebuttal",    "score_pro_rebuttal")
    workflow.add_edge("score_pro_rebuttal",   "debater2_rebuttal")
    workflow.add_edge("debater2_rebuttal",    "score_con_rebuttal")
    workflow.add_edge("score_con_rebuttal",   "debater1_rebuttal2")   # NEW
    workflow.add_edge("debater1_rebuttal2",   "score_pro_rebuttal2")  # NEW
    workflow.add_edge("score_pro_rebuttal2",  "debater2_rebuttal2")   # NEW
    workflow.add_edge("debater2_rebuttal2",   "score_con_rebuttal2")  # NEW
    workflow.add_edge("score_con_rebuttal2",  "convergence")          # updated
    workflow.add_edge("convergence",          "verdict")
    workflow.add_edge("verdict",              END)

    workflow.set_entry_point("moderator")
    return workflow.compile()


# =============================================================================
# Public Interface
# =============================================================================

def run_debate(
    topic: str,
    max_arg_iterations: int = 5,
    quality_model_path: str = "argument_quality_model_4features.pth",
) -> dict:
    """
    Run the full debate pipeline on the given topic.

    Parameters
    ----------
    topic               : The biomedical question to debate.
    max_arg_iterations  : Max refinement iterations per debater argument.
    quality_model_path  : Path to the argument quality .pth model file.
    """
    load_models(quality_model_path)
    initialize_runtime_models()

    initial: OrchestratorState = {
        "topic":               topic,
        "max_arg_iterations":  max_arg_iterations,
        "framed_topic":        "",
        "context_summary":     "",
        # PRO argument
        "d1_argument":           "",
        "d1_quality_score":      0.0,
        "d1_quality_level":      "",
        "d1_pubmed_query":       "",
        "d1_iterations_used":    0,
        "d1_retrieval_attempts": 0,
        "d1_accepted":           False,
        "d1_scores":             {},
        "d1_evidence":           [],
        # CON argument
        "d2_argument":           "",
        "d2_quality_score":      0.0,
        "d2_quality_level":      "",
        "d2_pubmed_query":       "",
        "d2_iterations_used":    0,
        "d2_retrieval_attempts": 0,
        "d2_accepted":           False,
        "d2_scores":             {},
        "d2_evidence":           [],
        # rebuttal round 1
        "d1_rebuttal":                "",
        "d1_rebuttal_logical_flaws":  [],
        "d1_rebuttal_counter_points": [],
        "d1_rebuttal_evidence":       {},
        "d2_rebuttal":                "",
        "d2_rebuttal_logical_flaws":  [],
        "d2_rebuttal_counter_points": [],
        "d2_rebuttal_evidence":       {},
        # rebuttal round 2
        "d1_rebuttal2":                "",
        "d1_rebuttal2_logical_flaws":  [],
        "d1_rebuttal2_counter_points": [],
        "d1_rebuttal2_evidence":       {},
        "d2_rebuttal2":                "",
        "d2_rebuttal2_logical_flaws":  [],
        "d2_rebuttal2_counter_points": [],
        "d2_rebuttal2_evidence":       {},
        # moderator scores (6 speeches × /10 = max 30 per side)
        "score_pro_arg":       0.0,
        "score_con_arg":       0.0,
        "score_pro_rebuttal":  0.0,
        "score_con_rebuttal":  0.0,
        "score_pro_rebuttal2": 0.0,
        "score_con_rebuttal2": 0.0,
        # convergence + verdict
        "convergence_summary": "",
        "winner":              "",
        "pro_nli_score":       0.0,
        "con_nli_score":       0.0,
        "pro_semantic_score":  0.0,
        "con_semantic_score":  0.0,
        "verdict_answer":      "",
        "verdict_brief":       "",
        "verdict_elaborate":   "",
        "messages":            [],
    }

    print("\n" + "=" * 80)
    print("🎯 FULL DEBATE ORCHESTRATOR  (2 rebuttal rounds per side)")
    print("=" * 80)
    print(f"Topic              : {topic}")
    print(f"Max Arg Iterations : {max_arg_iterations}")
    print(
        "\nPipeline:\n"
        "  moderator\n"
        "  → debater1_arg       → score_pro_arg\n"
        "  → debater2_arg       → score_con_arg\n"
        "  → debater1_rebuttal  → score_pro_rebuttal   [R1: PRO rebuts CON opening]\n"
        "  → debater2_rebuttal  → score_con_rebuttal   [R1: CON rebuts PRO opening]\n"
        "  → debater1_rebuttal2 → score_pro_rebuttal2  [R2: PRO rebuts CON R1]\n"
        "  → debater2_rebuttal2 → score_con_rebuttal2  [R2: CON rebuts PRO R1]\n"
        "  → convergence → verdict\n"
        f"\nModels:\n"
        f"  LLM  : mistralai/Mistral-7B-Instruct-v0.3 (local)\n"
        f"  NLI  : {_NLI_MODEL_ID}\n"
        f"  Embed: {_EMBED_MODEL_ID}\n"
    )

    final_state = create_orchestrator_graph().invoke(initial)

    pro_total = (
        final_state["score_pro_arg"]
        + final_state["score_pro_rebuttal"]
        + final_state["score_pro_rebuttal2"]
    )
    con_total = (
        final_state["score_con_arg"]
        + final_state["score_con_rebuttal"]
        + final_state["score_con_rebuttal2"]
    )

    result = {
        "topic":           final_state["topic"],
        "framed_topic":    final_state["framed_topic"],
        "context_summary": final_state["context_summary"],

        "pro": {
            "argument":           final_state["d1_argument"],
            "quality_score":      final_state["d1_quality_score"],
            "quality_level":      final_state["d1_quality_level"],
            "pubmed_query":       final_state["d1_pubmed_query"],
            "iterations_used":    final_state["d1_iterations_used"],
            "retrieval_attempts": final_state["d1_retrieval_attempts"],
            "accepted":           final_state["d1_accepted"],
            "scores":             final_state["d1_scores"],
            "evidence":           final_state["d1_evidence"],
        },
        "con": {
            "argument":           final_state["d2_argument"],
            "quality_score":      final_state["d2_quality_score"],
            "quality_level":      final_state["d2_quality_level"],
            "pubmed_query":       final_state["d2_pubmed_query"],
            "iterations_used":    final_state["d2_iterations_used"],
            "retrieval_attempts": final_state["d2_retrieval_attempts"],
            "accepted":           final_state["d2_accepted"],
            "scores":             final_state["d2_scores"],
            "evidence":           final_state["d2_evidence"],
        },

        "pro_rebuttal": {
            "rebuttal":        final_state["d1_rebuttal"],
            "logical_flaws":   final_state["d1_rebuttal_logical_flaws"],
            "counter_points":  final_state["d1_rebuttal_counter_points"],
            "evidence_counts": final_state["d1_rebuttal_evidence"],
        },
        "con_rebuttal": {
            "rebuttal":        final_state["d2_rebuttal"],
            "logical_flaws":   final_state["d2_rebuttal_logical_flaws"],
            "counter_points":  final_state["d2_rebuttal_counter_points"],
            "evidence_counts": final_state["d2_rebuttal_evidence"],
        },

        "pro_rebuttal2": {
            "rebuttal":        final_state["d1_rebuttal2"],
            "logical_flaws":   final_state["d1_rebuttal2_logical_flaws"],
            "counter_points":  final_state["d1_rebuttal2_counter_points"],
            "evidence_counts": final_state["d1_rebuttal2_evidence"],
        },
        "con_rebuttal2": {
            "rebuttal":        final_state["d2_rebuttal2"],
            "logical_flaws":   final_state["d2_rebuttal2_logical_flaws"],
            "counter_points":  final_state["d2_rebuttal2_counter_points"],
            "evidence_counts": final_state["d2_rebuttal2_evidence"],
        },

        "convergence_summary": final_state["convergence_summary"],
        "winner":              final_state["winner"],

        "moderator_scores": {
            "pro_arg":       final_state["score_pro_arg"],
            "con_arg":       final_state["score_con_arg"],
            "pro_rebuttal":  final_state["score_pro_rebuttal"],
            "con_rebuttal":  final_state["score_con_rebuttal"],
            "pro_rebuttal2": final_state["score_pro_rebuttal2"],
            "con_rebuttal2": final_state["score_con_rebuttal2"],
            "pro_total":     pro_total,
            "con_total":     con_total,
        },

        "local_scores": {
            "pro_nli":      final_state["pro_nli_score"],
            "con_nli":      final_state["con_nli_score"],
            "pro_semantic": final_state["pro_semantic_score"],
            "con_semantic": final_state["con_semantic_score"],
        },

        "verdict": {
            "answer":    final_state["verdict_answer"],
            "brief":     final_state["verdict_brief"],
            "elaborate": final_state["verdict_elaborate"],
        },

        "messages": final_state["messages"],

        # Flat evidence lists for downstream consumers (all rounds combined)
        "pro_evidence": (
            final_state["d1_evidence"]
            + [{"counter_point": k, "count": v}
               for k, v in final_state["d1_rebuttal_evidence"].items()]
            + [{"counter_point": k, "count": v}
               for k, v in final_state["d1_rebuttal2_evidence"].items()]
        ),
        "con_evidence": (
            final_state["d2_evidence"]
            + [{"counter_point": k, "count": v}
               for k, v in final_state["d2_rebuttal_evidence"].items()]
            + [{"counter_point": k, "count": v}
               for k, v in final_state["d2_rebuttal2_evidence"].items()]
        ),
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✅ DEBATE COMPLETE")
    print("=" * 80)
    print(f"Motion : {result['framed_topic']}")

    print(f"\n🔵 PRO  quality={result['pro']['quality_score']:.3f}"
          f"  [{result['pro']['quality_level']}]  accepted={result['pro']['accepted']}")
    print(f"   NLI={result['local_scores']['pro_nli']:.3f}  "
          f"Semantic={result['local_scores']['pro_semantic']:.3f}  "
          f"Evidence={len(result['pro']['evidence'])} piece(s)")

    print(f"\n🔴 CON  quality={result['con']['quality_score']:.3f}"
          f"  [{result['con']['quality_level']}]  accepted={result['con']['accepted']}")
    print(f"   NLI={result['local_scores']['con_nli']:.3f}  "
          f"Semantic={result['local_scores']['con_semantic']:.3f}  "
          f"Evidence={len(result['con']['evidence'])} piece(s)")

    ms = result["moderator_scores"]
    print(f"\n📊 Moderator Scorecard:")
    print(f"   PRO argument      : {ms['pro_arg']:.1f}/10")
    print(f"   CON argument      : {ms['con_arg']:.1f}/10")
    print(f"   PRO rebuttal R1   : {ms['pro_rebuttal']:.1f}/10")
    print(f"   CON rebuttal R1   : {ms['con_rebuttal']:.1f}/10")
    print(f"   PRO rebuttal R2   : {ms['pro_rebuttal2']:.1f}/10")
    print(f"   CON rebuttal R2   : {ms['con_rebuttal2']:.1f}/10")
    print(f"   ─────────────────────────────────")
    print(f"   PRO total         : {ms['pro_total']:.1f}/30")
    print(f"   CON total         : {ms['con_total']:.1f}/30")

    print(f"\n⚖️  Winner     : {result['winner']}")
    print(f"📋 Convergence: {result['convergence_summary']}")

    print(f"\n{'='*80}")
    print(f"🏛️  FINAL VERDICT : {result['verdict']['answer']}")
    print(f"📝 Brief          : {result['verdict']['brief']}")
    print(f"\n📖 Elaborate:\n{result['verdict']['elaborate']}")
    print("=" * 80)

    return result


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    result = run_debate(
        topic=(
            "Are group 2 innate lymphoid cells (ILC2s) increased in "
            "chronic rhinosinusitis with nasal polyps or eosinophilia?"
        ),
        max_arg_iterations=5,
    )
    for key, value in result.items():
        print(f"\n{key}:\n{value}")
        print("=" * 80)