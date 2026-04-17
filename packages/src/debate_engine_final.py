"""
Full Debate Orchestrator Graph (LangGraph)
==========================================

Pipeline:
  start
    │
    ▼
  moderator_node          ← sets topic, extracts context, frames the debate
    │
    ▼
  debater1_arg_node       ← Debater 1 (PRO)  → debate_argument_generator subgraph
    │
    ▼
  debater2_arg_node       ← Debater 2 (CON)  → debate_argument_generator subgraph
    │
    ▼
  debater1_rebuttal_node  ← Debater 1 rebuts CON argument → rebuttal subgraph
    │
    ▼
  debater2_rebuttal_node  ← Debater 2 rebuts PRO argument → rebuttal subgraph
    │
    ▼
  convergence_node        ← Moderator summarises + declares outcome
    │
    ▼
  verdict_node            ← Final YES/NO answer + elaborate explanation
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

    # debater 1 (PRO) – rebuttal
    d1_rebuttal:                str
    d1_rebuttal_logical_flaws:  list
    d1_rebuttal_counter_points: list
    d1_rebuttal_evidence:       dict

    # debater 2 (CON) – rebuttal
    d2_rebuttal:                str
    d2_rebuttal_logical_flaws:  list
    d2_rebuttal_counter_points: list
    d2_rebuttal_evidence:       dict

    # moderator scores out of 10 — assigned after each of the 4 rounds
    score_pro_arg:      float   # PRO opening argument
    score_con_arg:      float   # CON opening argument
    score_pro_rebuttal: float   # PRO rebuttal
    score_con_rebuttal: float   # CON rebuttal

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
    Call llm_inference with the conversation messages.

    llm_inference is assumed to accept either:
      (a) a pre-formatted prompt string, OR
      (b) a list of {"role": ..., "content": ...} dicts.

    We try (b) first; if the function signature only accepts a string we
    fall back to (a) with minimal-instruct formatting.

    IMPORTANT: We deliberately do NOT double-wrap with <s>[INST]…[/INST]
    if llm_inference already applies its own chat template internally.
    Inspect get_judge_lm.py if you still get empty responses — the template
    may need to be removed from one side.
    """
    import inspect
    sig = inspect.signature(llm_inference)
    params = list(sig.parameters.keys())

    # Build a plain user string as fallback
    user_parts = [m["content"] for m in messages if m.get("role") == "user"]
    plain_prompt = "\n\n".join(user_parts)

    # Prefer passing the messages list directly when the function accepts it
    first_param = params[0] if params else "prompt"
    if first_param in ("messages", "conversation"):
        raw = llm_inference(messages, max_new_tokens=max_new_tokens)
    else:
        # Single-string path — use minimal instruct format
        # Only wrap if llm_inference does NOT apply its own template.
        # If you see double [INST] tags in logs, remove the wrapping here.
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

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the first {...} block
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Try to repair common LLM mistakes: single quotes, trailing commas
    repaired = cleaned
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)       # trailing commas
    repaired = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', repaired)  # single → double quotes
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

    On each retry:
      - Waits backoff_seconds * attempt before retrying (linear backoff)
      - Appends the failed raw response + a correction nudge to the
        conversation so the model can see its own mistake and self-correct
      - Validates that all required_keys are present in the parsed result

    Returns
    -------
    (parsed_dict, raw_string)  on success
    Raises the last exception if all retries are exhausted.
    """
    import time

    last_exc: Exception = RuntimeError("No attempts made.")
    last_raw: str = ""
    conversation = list(messages)  # copy so we don't mutate the caller's list

    for attempt in range(1, max_retries + 1):
        try:
            raw       = _local_chat(conversation, max_new_tokens=max_new_tokens)
            last_raw  = raw
            parsed    = _extract_json(raw)

            # Validate all required keys exist and are non-empty strings
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

                # Append the bad response + correction nudge so the model
                # can see its own mistake on the next attempt
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
    """
    Derive winner from automated scores when the LLM fails.

    NOTE: scores keys must match _score_argument_pair output:
      pro_nli_score, con_nli_score, pro_semantic_score, con_semantic_score
    """
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
        # Use default hypothesis_template so bart-large-mnli gets
        # "This example is {label}." — passing "{}" breaks the model.
        result = _nli_pipeline(
            premise[:1024],
            candidate_labels=[hypothesis[:200]],
            multi_label=False,
        )
        # zero-shot-classification returns scores for each label;
        # for a single label the entailment score is result["scores"][0]
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
        "opposition_arg":     "",   # PRO speaks first — no opposition arg yet
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
# Node 3 – Debater 2 (CON) – Argument
# =============================================================================

def debater2_arg_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔴 DEBATER 2 (CON) – Generating opening argument")
    print(f"{'='*70}")

    # CON is intentionally given PRO's argument as context (second-speaker advantage)
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
        "opposition_arg":     state["d1_argument"],
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
# Node 4 – Debater 1 (PRO) – Rebuttal
# =============================================================================

def debater1_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔵 DEBATER 1 (PRO) – Rebutting CON argument")
    print(f"{'='*70}")

    initial: RebuttalState = {
        "original_argument": state["d2_argument"],
        "topic":             state["framed_topic"],
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
    print(f"\n   ✅ PRO Rebuttal:\n   {final['rebuttal'][:200]}{suffix}")
    print(f"   🔍 Logical flaws   : {len(final['logical_flaws'])}")
    print(f"   💡 Counter-points  : {len(final['counter_points'])}")
    print(f"   📚 PubMed abstracts: {sum(evidence_counts.values())}\n")

    return {
        "d1_rebuttal":                final["rebuttal"],
        "d1_rebuttal_logical_flaws":  final["logical_flaws"],
        "d1_rebuttal_counter_points": final["counter_points"],
        "d1_rebuttal_evidence":       evidence_counts,
        "messages": [{"role": "debater1_rebuttal", "content": final["rebuttal"]}],
    }


# =============================================================================
# Node 5 – Debater 2 (CON) – Rebuttal
# =============================================================================

def debater2_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔴 DEBATER 2 (CON) – Rebutting PRO argument")
    print(f"{'='*70}")

    initial: RebuttalState = {
        "original_argument": state["d1_argument"],
        "topic":             state["framed_topic"],
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
    print(f"\n   ✅ CON Rebuttal:\n   {final['rebuttal'][:200]}{suffix}")
    print(f"   🔍 Logical flaws   : {len(final['logical_flaws'])}")
    print(f"   💡 Counter-points  : {len(final['counter_points'])}")
    print(f"   📚 PubMed abstracts: {sum(evidence_counts.values())}\n")

    return {
        "d2_rebuttal":                final["rebuttal"],
        "d2_rebuttal_logical_flaws":  final["logical_flaws"],
        "d2_rebuttal_counter_points": final["counter_points"],
        "d2_rebuttal_evidence":       evidence_counts,
        "messages": [{"role": "debater2_rebuttal", "content": final["rebuttal"]}],
    }


# =============================================================================
# Scoring helper — moderator scores a single argument or rebuttal out of 10
# =============================================================================

def _moderator_score(role: str, text: str, topic: str, context: str = "") -> float:
    """
    Ask the moderator to score a piece of debate text out of 10.

    Extracts a plain integer/float from the response — no JSON required.
    Falls back to 5.0 if nothing numeric is found.

    Parameters
    ----------
    role    : "PRO argument" | "CON argument" | "PRO rebuttal" | "CON rebuttal"
    text    : The argument or rebuttal text to score.
    topic   : The framed debate motion.
    context : Optional context (e.g. the opposing argument for rebuttal scoring).
    """
    prompt = (
        f"You are a strict debate judge scoring a {role} on the motion:\n"
        f"\"{topic}\"\n\n"
    )
    if context:
        prompt += f"For reference, the opposing argument was:\n{_trunc(context, 300)}\n\n"
    prompt += (
        f"{role.upper()}:\n{_trunc(text, 500)}\n\n"
        "Score this {role} out of 10 based on:\n"
        "  • Strength and relevance of evidence\n"
        "  • Logical coherence\n"
        "  • Directly addressing the motion\n"
        "  • Clarity and persuasiveness\n\n"
        "Reply with ONLY a single number between 0 and 10 (decimals allowed). "
        "No explanation, no text — just the number."
    )

    raw = _local_chat([{"role": "user", "content": prompt}], max_new_tokens=10)

    # Extract the first number we find in the response
    match = re.search(r"\b(\d{1,2}(?:\.\d{1,2})?)\b", raw or "")
    if match:
        score = float(match.group(1))
        score = max(0.0, min(10.0, score))   # clamp to [0, 10]
        print(f"   🎯 Moderator score for {role}: {score:.1f}/10")
        return score

    print(f"   ⚠  Could not parse score from: {raw!r} — defaulting to 5.0")
    return 5.0


# =============================================================================
# Node 6a – Moderator scores PRO opening argument
# =============================================================================

def score_pro_arg_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – PRO Opening Argument")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "PRO argument",
        text    = state["d1_argument"],
        topic   = state["framed_topic"],
    )
    return {
        "score_pro_arg": score,
        "messages": [{"role": "moderator_score", "content": f"PRO argument score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 6b – Moderator scores CON opening argument
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
        "messages": [{"role": "moderator_score", "content": f"CON argument score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 6c – Moderator scores PRO rebuttal
# =============================================================================

def score_pro_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – PRO Rebuttal")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "PRO rebuttal",
        text    = state["d1_rebuttal"],
        topic   = state["framed_topic"],
        context = state["d2_argument"],
    )
    return {
        "score_pro_rebuttal": score,
        "messages": [{"role": "moderator_score", "content": f"PRO rebuttal score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 6d – Moderator scores CON rebuttal
# =============================================================================

def score_con_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR SCORING – CON Rebuttal")
    print(f"{'='*70}")
    score = _moderator_score(
        role    = "CON rebuttal",
        text    = state["d2_rebuttal"],
        topic   = state["framed_topic"],
        context = state["d1_argument"],
    )
    return {
        "score_con_rebuttal": score,
        "messages": [{"role": "moderator_score", "content": f"CON rebuttal score: {score:.1f}/10"}],
    }


# =============================================================================
# Node 6 – Convergence  (winner decided by scores — NO JSON parsing)
# =============================================================================

def convergence_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("⚖️  CONVERGENCE NODE – Tallying scores and generating summary")
    print(f"{'='*70}")

    # ── Tally scores ──────────────────────────────────────────────────────────
    pro_total = state["score_pro_arg"] + state["score_pro_rebuttal"]
    con_total = state["score_con_arg"] + state["score_con_rebuttal"]

    if pro_total > con_total:
        winner = "PRO"
    elif con_total > pro_total:
        winner = "CON"
    else:
        winner = "DRAW"

    print(f"\n   📊 Scorecard:")
    print(f"      PRO argument  : {state['score_pro_arg']:.1f}/10")
    print(f"      CON argument  : {state['score_con_arg']:.1f}/10")
    print(f"      PRO rebuttal  : {state['score_pro_rebuttal']:.1f}/10")
    print(f"      CON rebuttal  : {state['score_con_rebuttal']:.1f}/10")
    print(f"      ─────────────────────")
    print(f"      PRO total     : {pro_total:.1f}/20")
    print(f"      CON total     : {con_total:.1f}/20")
    print(f"   🏆 Winner        : {winner}")

    # ── Run local NLI + semantic scoring (kept for verdict node context) ──────
    scores  = _score_argument_pair(
        framed_topic=state["framed_topic"],
        pro_argument=state["d1_argument"],
        con_argument=state["d2_argument"],
    )

    # ── Generate plain-text summary — no JSON, just ask for prose ─────────────
    summary_prompt = (
        "You are a professional debate moderator writing a closing summary.\n\n"
        f"Motion: {state['framed_topic']}\n\n"
        f"PRO argument (scored {state['score_pro_arg']:.1f}/10):\n"
        f"{_trunc(state['d1_argument'], 400)}\n\n"
        f"CON argument (scored {state['score_con_arg']:.1f}/10):\n"
        f"{_trunc(state['d2_argument'], 400)}\n\n"
        f"PRO rebuttal (scored {state['score_pro_rebuttal']:.1f}/10):\n"
        f"{_trunc(state['d1_rebuttal'], 300)}\n\n"
        f"CON rebuttal (scored {state['score_con_rebuttal']:.1f}/10):\n"
        f"{_trunc(state['d2_rebuttal'], 300)}\n\n"
        f"Final scores — PRO: {pro_total:.1f}/20  |  CON: {con_total:.1f}/20\n"
        f"Winner: {winner}\n\n"
        "Write a 3-5 sentence closing summary of the debate highlighting the "
        "key clash points and why the scores fell the way they did. "
        "Plain prose only — no JSON, no bullet points."
    )

    raw_summary = _local_chat([{"role": "user", "content": summary_prompt}], max_new_tokens=300)
    # Use whatever the model returned as-is — no parsing needed
    conv_summary = raw_summary.strip() if raw_summary.strip() else (
        f"The debate on '{state['framed_topic']}' concluded with "
        f"PRO scoring {pro_total:.1f}/20 and CON scoring {con_total:.1f}/20. "
        f"{winner} is declared the winner."
    )

    display_summary = (
        f"{conv_summary}\n\n"
        f"📊 Scores — PRO: {pro_total:.1f}/20  |  CON: {con_total:.1f}/20\n"
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
# Node 7 – Verdict
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

    pro_ev     = _fmt_evidence(state.get("d1_evidence", []))
    con_ev     = _fmt_evidence(state.get("d2_evidence", []))
    pro_reb_ev = _fmt_rebuttal_ev(state.get("d1_rebuttal_evidence", {}))
    con_reb_ev = _fmt_rebuttal_ev(state.get("d2_rebuttal_evidence", {}))

    pro_nli = state.get("pro_nli_score", 0.0)
    con_nli = state.get("con_nli_score", 0.0)
    pro_sem = state.get("pro_semantic_score", 0.0)
    con_sem = state.get("con_semantic_score", 0.0)

    prompt = (
        "You are an expert biomedical analyst who has observed a structured debate.\n"
        "Deliver a definitive, evidence-grounded verdict on the original question.\n\n"
        f"ORIGINAL QUESTION: {state['topic']}\n"
        f"FRAMED MOTION: {state['framed_topic']}\n"
        f"BACKGROUND: {_trunc(state['context_summary'], 300)}\n\n"
        f"PRO ARGUMENT (quality {state['d1_quality_score']:.3f}):\n"
        f"{_trunc(state['d1_argument'], 400)}\n"
        f"PRO PubMed evidence:\n{pro_ev}\n\n"
        f"CON ARGUMENT (quality {state['d2_quality_score']:.3f}):\n"
        f"{_trunc(state['d2_argument'], 400)}\n"
        f"CON PubMed evidence:\n{con_ev}\n\n"
        f"PRO REBUTTAL:\n{_trunc(state['d1_rebuttal'], 300)}\n"
        f"PRO rebuttal evidence:\n{pro_reb_ev}\n\n"
        f"CON REBUTTAL:\n{_trunc(state['d2_rebuttal'], 300)}\n"
        f"CON rebuttal evidence:\n{con_reb_ev}\n\n"
        f"LOCAL SCORES — PRO NLI: {pro_nli:.3f}, PRO Semantic: {pro_sem:.3f}, "
        f"CON NLI: {con_nli:.3f}, CON Semantic: {con_sem:.3f}\n\n"
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

    workflow.add_node("moderator",           moderator_node)
    workflow.add_node("debater1_arg",        debater1_arg_node)
    workflow.add_node("score_pro_arg",       score_pro_arg_node)
    workflow.add_node("debater2_arg",        debater2_arg_node)
    workflow.add_node("score_con_arg",       score_con_arg_node)
    workflow.add_node("debater1_rebuttal",   debater1_rebuttal_node)
    workflow.add_node("score_pro_rebuttal",  score_pro_rebuttal_node)
    workflow.add_node("debater2_rebuttal",   debater2_rebuttal_node)
    workflow.add_node("score_con_rebuttal",  score_con_rebuttal_node)
    workflow.add_node("convergence",         convergence_node)
    workflow.add_node("verdict",             verdict_node)

    workflow.add_edge("moderator",          "debater1_arg")
    workflow.add_edge("debater1_arg",       "score_pro_arg")       # score PRO arg
    workflow.add_edge("score_pro_arg",      "debater2_arg")
    workflow.add_edge("debater2_arg",       "score_con_arg")       # score CON arg
    workflow.add_edge("score_con_arg",      "debater1_rebuttal")
    workflow.add_edge("debater1_rebuttal",  "score_pro_rebuttal")  # score PRO rebuttal
    workflow.add_edge("score_pro_rebuttal", "debater2_rebuttal")
    workflow.add_edge("debater2_rebuttal",  "score_con_rebuttal")  # score CON rebuttal
    workflow.add_edge("score_con_rebuttal", "convergence")
    workflow.add_edge("convergence",        "verdict")
    workflow.add_edge("verdict",            END)

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
    load_models(quality_model_path)       # argument quality scorer
    initialize_runtime_models()           # NLI + embedding models

    initial: OrchestratorState = {
        "topic":               topic,
        "max_arg_iterations":  max_arg_iterations,
        "framed_topic":        "",
        "context_summary":     "",
        "d1_argument":           "",
        "d1_quality_score":      0.0,
        "d1_quality_level":      "",
        "d1_pubmed_query":       "",
        "d1_iterations_used":    0,
        "d1_retrieval_attempts": 0,
        "d1_accepted":           False,
        "d1_scores":             {},
        "d1_evidence":           [],
        "d2_argument":           "",
        "d2_quality_score":      0.0,
        "d2_quality_level":      "",
        "d2_pubmed_query":       "",
        "d2_iterations_used":    0,
        "d2_retrieval_attempts": 0,
        "d2_accepted":           False,
        "d2_scores":             {},
        "d2_evidence":           [],
        "d1_rebuttal":                "",
        "d1_rebuttal_logical_flaws":  [],
        "d1_rebuttal_counter_points": [],
        "d1_rebuttal_evidence":       {},
        "d2_rebuttal":                "",
        "d2_rebuttal_logical_flaws":  [],
        "d2_rebuttal_counter_points": [],
        "d2_rebuttal_evidence":       {},
        "score_pro_arg":      0.0,
        "score_con_arg":      0.0,
        "score_pro_rebuttal": 0.0,
        "score_con_rebuttal": 0.0,
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
    print("🎯 FULL DEBATE ORCHESTRATOR")
    print("=" * 80)
    print(f"Topic              : {topic}")
    print(f"Max Arg Iterations : {max_arg_iterations}")
    print(
        "\nPipeline:\n"
        "  moderator → debater1_arg → score_pro_arg\n"
        "           → debater2_arg → score_con_arg\n"
        "           → debater1_rebuttal → score_pro_rebuttal\n"
        "           → debater2_rebuttal → score_con_rebuttal\n"
        "           → convergence → verdict\n"
        f"\nModels:\n"
        f"  LLM  : mistralai/Mistral-7B-Instruct-v0.3 (local)\n"
        f"  NLI  : {_NLI_MODEL_ID}\n"
        f"  Embed: {_EMBED_MODEL_ID}\n"
    )

    final_state = create_orchestrator_graph().invoke(initial)

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

        "convergence_summary": final_state["convergence_summary"],
        "winner":              final_state["winner"],

        "moderator_scores": {
            "pro_arg":      final_state["score_pro_arg"],
            "con_arg":      final_state["score_con_arg"],
            "pro_rebuttal": final_state["score_pro_rebuttal"],
            "con_rebuttal": final_state["score_con_rebuttal"],
            "pro_total":    final_state["score_pro_arg"] + final_state["score_pro_rebuttal"],
            "con_total":    final_state["score_con_arg"] + final_state["score_con_rebuttal"],
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

        # Convenience: flat evidence lists for downstream consumers
        "pro_evidence": (
            final_state["d1_evidence"] + [
                {"counter_point": k, "count": v}
                for k, v in final_state["d1_rebuttal_evidence"].items()
            ]
        ),
        "con_evidence": (
            final_state["d2_evidence"] + [
                {"counter_point": k, "count": v}
                for k, v in final_state["d2_rebuttal_evidence"].items()
            ]
        ),
    }

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
    print(f"   PRO argument  : {ms['pro_arg']:.1f}/10")
    print(f"   CON argument  : {ms['con_arg']:.1f}/10")
    print(f"   PRO rebuttal  : {ms['pro_rebuttal']:.1f}/10")
    print(f"   CON rebuttal  : {ms['con_rebuttal']:.1f}/10")
    print(f"   ─────────────────────────────")
    print(f"   PRO total     : {ms['pro_total']:.1f}/20")
    print(f"   CON total     : {ms['con_total']:.1f}/20")
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