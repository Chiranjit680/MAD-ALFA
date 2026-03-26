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

Model usage
-----------
  Llama-3-8B-Instruct               : HuggingFace Inference API (InferenceClient)
                                      — all LLM text generation
  facebook/bart-large-mnli          : local .model_store — NLI entailment scoring
  sentence-transformers/all-MiniLM-L6-v2 : local .model_store — semantic similarity

All local models are initialised lazily on first use with a VRAM pre-flight
check and automatic CPU fallback (same pattern as rebuttal_node.py).

Evidence captured at every stage and surfaced in the final result:
  result["pro"]["evidence"]                – reranked_evidence list from PRO arg subgraph
  result["pro"]["scores"]                  – nli / semantic / confidence / overall
  result["con"]["evidence"]                – reranked_evidence list from CON arg subgraph
  result["con"]["scores"]                  – nli / semantic / confidence / overall
  result["pro_rebuttal"]["logical_flaws"]  – flaws found in CON argument
  result["pro_rebuttal"]["counter_points"] – counter-points raised by PRO
  result["pro_rebuttal"]["evidence_counts"]– {counter_point: abstract_count}
  result["con_rebuttal"]["logical_flaws"]  – flaws found in PRO argument
  result["con_rebuttal"]["counter_points"] – counter-points raised by CON
  result["con_rebuttal"]["evidence_counts"]– {counter_point: abstract_count}
  result["local_scores"]                   – pro/con NLI + semantic scores
"""

from __future__ import annotations

import json
import os
import re
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

import torch
from langgraph.graph import StateGraph, END
from huggingface_hub import InferenceClient
from transformers import pipeline as hf_pipeline

try:
    from .local_model_store import get_local_model_dir
except ImportError:
    from local_model_store import get_local_model_dir

from debate_agent import DebateState, create_debate_argument_graph
from rebuttal_node import RebuttalState, create_rebuttal_graph


load_dotenv()  # Load environment variables from .env file if present
# =============================================================================
# Runtime model configuration
# =============================================================================

_HAS_CUDA = torch.cuda.is_available()
_DEVICE_ID = 0 if _HAS_CUDA else -1  # Use first available GPU or CPU
_DTYPE = torch.float16 if _HAS_CUDA else torch.float32

# Model IDs — must match directory names under .model_store
_NLI_MODEL_ID   = "facebook/bart-large-mnli"
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy handles — populated by initialize_runtime_models()
_nli_pipeline   = None   # zero-shot-classification  (bart-large-mnli)
_embed_pipeline = None   # feature-extraction         (all-MiniLM-L6-v2)


def _load_pipeline_with_fallback(
    task: str,
    model_path: str,
    device_id: int,
    dtype: torch.dtype,
    **kwargs,
):
    """Load a transformers pipeline; retry on CPU if a GPU OOM occurs."""
    try:
        return hf_pipeline(
            task,
            model=model_path,
            device=device_id,
            torch_dtype=dtype,
            **kwargs,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device_id != -1:
            print(f"   ⚠️  OOM loading {model_path} on GPU — "
                  f"clearing cache and retrying on CPU.")
            torch.cuda.empty_cache()
            return hf_pipeline(
                task,
                model=model_path,
                device=-1,
                torch_dtype=torch.float32,
                **kwargs,
            )
        raise


# =============================================================================
# Lazy model initialisation
# =============================================================================

def initialize_runtime_models(force_reload: bool = False) -> None:
    """
    Lazily load all local models needed by the orchestrator.
    Safe to call multiple times — skips re-init unless force_reload=True.
    """
    global _nli_pipeline, _embed_pipeline

    if not force_reload and _nli_pipeline is not None and _embed_pipeline is not None:
        return

    print("\n🚀 Initializing debate engine runtime models...")
    print(f"   CUDA available : {_HAS_CUDA}")
    if _HAS_CUDA:
        print(f"   GPU device ID  : {_DEVICE_ID}")
        print(f"   GPU name       : {torch.cuda.get_device_name(_DEVICE_ID)}")

    # ── NLI model (bart-large-mnli) ──────────────────────────────────────────
    if force_reload or _nli_pipeline is None:
        print(f"   Loading NLI    : {_NLI_MODEL_ID}  "
              f"({'CPU' if _DEVICE_ID == -1 else f'GPU:{_DEVICE_ID}'}, {_DTYPE})")
        nli_path = get_local_model_dir(_NLI_MODEL_ID)
        _nli_pipeline = _load_pipeline_with_fallback(
            "zero-shot-classification",
            nli_path,
            _DEVICE_ID,
            _DTYPE,
        )
        print("   ✅ NLI model ready")

    # ── Embedding model (all-MiniLM-L6-v2) ───────────────────────────────────
    # Uses same GPU device as NLI model for fast inference.
    if force_reload or _embed_pipeline is None:
        print(f"   Loading Embed  : {_EMBED_MODEL_ID}  "
              f"({'CPU' if _DEVICE_ID == -1 else f'GPU:{_DEVICE_ID}'}, {_DTYPE})")
        embed_path = get_local_model_dir(_EMBED_MODEL_ID)
        _embed_pipeline = _load_pipeline_with_fallback(
            "feature-extraction",
            embed_path,
            _DEVICE_ID,
            _DTYPE,
            return_tensor=False,
        )
        print("   ✅ Embedding model ready")

    print("✅ All debate engine runtime models loaded.\n")


# =============================================================================
# Orchestrator State
# =============================================================================

class OrchestratorState(TypedDict):
    # ── inputs ────────────────────────────────────────────────────────────────
    topic: str
    max_arg_iterations: int

    # ── moderator outputs ─────────────────────────────────────────────────────
    framed_topic: str
    context_summary: str

    # ── debater 1 (PRO) – argument ────────────────────────────────────────────
    d1_argument: str
    d1_quality_score: float
    d1_quality_level: str
    d1_pubmed_query: str
    d1_iterations_used: int
    d1_retrieval_attempts: int
    d1_accepted: bool
    d1_scores: dict                  # {nli, semantic, confidence, overall}
    d1_evidence: list                # reranked_evidence from subgraph

    # ── debater 2 (CON) – argument ────────────────────────────────────────────
    d2_argument: str
    d2_quality_score: float
    d2_quality_level: str
    d2_pubmed_query: str
    d2_iterations_used: int
    d2_retrieval_attempts: int
    d2_accepted: bool
    d2_scores: dict
    d2_evidence: list

    # ── debater 1 (PRO) – rebuttal ────────────────────────────────────────────
    d1_rebuttal: str
    d1_rebuttal_logical_flaws: list
    d1_rebuttal_counter_points: list
    d1_rebuttal_evidence: dict       # {counter_point: abstract_count}

    # ── debater 2 (CON) – rebuttal ────────────────────────────────────────────
    d2_rebuttal: str
    d2_rebuttal_logical_flaws: list
    d2_rebuttal_counter_points: list
    d2_rebuttal_evidence: dict

    # ── convergence ───────────────────────────────────────────────────────────
    convergence_summary: str
    winner: str                      # "PRO" | "CON" | "DRAW"

    # ── local-model scores (set by convergence_node) ──────────────────────────
    pro_nli_score: float             # bart-large-mnli entailment: PRO arg → motion
    con_nli_score: float             # bart-large-mnli entailment: CON arg → motion
    pro_semantic_score: float        # cosine sim: PRO arg vs framed_topic
    con_semantic_score: float        # cosine sim: CON arg vs framed_topic

    # ── verdict ───────────────────────────────────────────────────────────────
    verdict_answer: str              # "YES" | "NO" | "INCONCLUSIVE"
    verdict_brief: str               # One-sentence direct answer
    verdict_elaborate: str           # Full elaborated explanation

    # ── trace ─────────────────────────────────────────────────────────────────
    messages: Annotated[list, operator.add]


# =============================================================================
# Llama-3 helper  —  HuggingFace Inference API
# =============================================================================

def _llama(messages: list[dict], max_tokens: int = 600) -> str:
    """Call Llama-3-8B-Instruct via the HuggingFace Inference API."""
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is not set.")
    print(f"   📡 Calling Llama-3-8B-Instruct via HF API (max_tokens={max_tokens}) …")
    client = InferenceClient(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=api_key,
        timeout=120,
    )
    resp = client.chat_completion(messages, max_tokens=max_tokens)
    print("   ✅ Response received.")
    return (resp.choices[0].message.content or "").strip()


def _extract_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON; falls back to brace extraction."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start: end + 1])
            except json.JSONDecodeError as inner:
                raise inner from exc
        raise


# =============================================================================
# Local-model scoring helpers
# =============================================================================

def _nli_score(premise: str, hypothesis: str) -> float:
    """
    Return the entailment probability of (premise → hypothesis) using
    bart-large-mnli from the local model store.  Returns 0.0 on any error.
    """
    if _nli_pipeline is None:
        initialize_runtime_models()
    try:
        result = _nli_pipeline(
            premise[:1024],          # bart hard limit
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="{}",
            multi_label=False,
        )
        label_scores = dict(zip(result["labels"], result["scores"]))
        return float(label_scores.get("entailment", 0.0))
    except Exception as exc:
        print(f"   ⚠️  NLI scoring error: {exc}")
        return 0.0


def _mean_pooling(token_embeddings: list) -> list[float]:
    """Average pool token embeddings → single sentence vector."""
    import numpy as np
    arr = np.array(token_embeddings)
    if arr.ndim == 3:
        arr = arr[0]     # [1, seq_len, hidden] → [seq_len, hidden]
    return arr.mean(axis=0).tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom != 0 else 0.0


def _semantic_score(text_a: str, text_b: str) -> float:
    """
    Return cosine similarity between two texts using all-MiniLM-L6-v2
    from the local model store.  Returns 0.0 on any error.
    """
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
    """
    Run NLI + semantic scoring for both arguments against the framed topic.
    Returns a dict with four float scores consumed by convergence and verdict.
    """
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
# Node 1 – Moderator  (Llama-3 via HF API)
# =============================================================================

def moderator_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🎙️  MODERATOR NODE – Framing the debate")
    print(f"{'='*70}")

    prompt = f"""You are a professional debate moderator.

Given the following debate topic, your tasks are:
1. Frame it as a clear, neutral debate motion (one sentence).
2. Write a 2-3 sentence background summary that contextualises the topic
   without taking sides.

Topic: {state["topic"]}

Respond ONLY with valid JSON, no explanation:
{{
  "framed_topic": "...",
  "context_summary": "..."
}}"""

    parsed          = _extract_json(_llama([{"role": "user", "content": prompt}], max_tokens=300))
    framed_topic    = parsed["framed_topic"]
    context_summary = parsed["context_summary"]

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

    final = create_debate_argument_graph().invoke(initial)

    suffix = "…" if len(final["argument"]) > 200 else ""
    print(f"\n   ✅ PRO Argument (quality={final['quality_score']:.3f}):\n"
          f"   {final['argument'][:200]}{suffix}")
    print(f"   📚 Evidence pieces retrieved : {len(final['reranked_evidence'])}\n")

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
        "opposition_arg":     state["d1_argument"],   # CON sees PRO's argument
        "messages":           [],
        "retrieval_feedback": "",
        "retrieval_attempts": 0,
        "failure_reason":     "",
    }

    final = create_debate_argument_graph().invoke(initial)

    suffix = "…" if len(final["argument"]) > 200 else ""
    print(f"\n   ✅ CON Argument (quality={final['quality_score']:.3f}):\n"
          f"   {final['argument'][:200]}{suffix}")
    print(f"   📚 Evidence pieces retrieved : {len(final['reranked_evidence'])}\n")

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
# Node 4 – Debater 1 (PRO) – Rebuttal of CON argument
# =============================================================================

def debater1_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔵 DEBATER 1 (PRO) – Rebutting CON argument")
    print(f"{'='*70}")

    initial: RebuttalState = {
        "original_argument": state["d2_argument"],   # PRO rebuts CON
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
    print(f"   🔍 Logical flaws identified  : {len(final['logical_flaws'])}")
    print(f"   💡 Counter-points raised     : {len(final['counter_points'])}")
    print(f"   📚 Total PubMed abstracts    : {sum(evidence_counts.values())}\n")

    return {
        "d1_rebuttal":                final["rebuttal"],
        "d1_rebuttal_logical_flaws":  final["logical_flaws"],
        "d1_rebuttal_counter_points": final["counter_points"],
        "d1_rebuttal_evidence":       evidence_counts,
        "messages": [{"role": "debater1_rebuttal", "content": final["rebuttal"]}],
    }


# =============================================================================
# Node 5 – Debater 2 (CON) – Rebuttal of PRO argument
# =============================================================================

def debater2_rebuttal_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("🔴 DEBATER 2 (CON) – Rebutting PRO argument")
    print(f"{'='*70}")

    initial: RebuttalState = {
        "original_argument": state["d1_argument"],   # CON rebuts PRO
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
    print(f"   🔍 Logical flaws identified  : {len(final['logical_flaws'])}")
    print(f"   💡 Counter-points raised     : {len(final['counter_points'])}")
    print(f"   📚 Total PubMed abstracts    : {sum(evidence_counts.values())}\n")

    return {
        "d2_rebuttal":                final["rebuttal"],
        "d2_rebuttal_logical_flaws":  final["logical_flaws"],
        "d2_rebuttal_counter_points": final["counter_points"],
        "d2_rebuttal_evidence":       evidence_counts,
        "messages": [{"role": "debater2_rebuttal", "content": final["rebuttal"]}],
    }


# =============================================================================
# Node 6 – Convergence  (local NLI + semantic  +  Llama-3 via HF API)
# =============================================================================

def convergence_node(state: OrchestratorState) -> dict:
    print(f"\n{'='*70}")
    print("⚖️  CONVERGENCE NODE – Moderator closes the debate")
    print(f"{'='*70}")

    # ── Step 1: score both arguments with local models ────────────────────────
    scores  = _score_argument_pair(
        framed_topic=state["framed_topic"],
        pro_argument=state["d1_argument"],
        con_argument=state["d2_argument"],
    )
    pro_nli = scores["pro_nli_score"]
    con_nli = scores["con_nli_score"]
    pro_sem = scores["pro_semantic_score"]
    con_sem = scores["con_semantic_score"]

    # ── Step 2: Llama-3 (HF API) writes the narrative summary ────────────────
    prompt = f"""You are a professional debate moderator closing the debate.

Motion: {state['framed_topic']}

─── PRO Opening Argument (quality score: {state['d1_quality_score']:.3f}) ───
{state['d1_argument']}

─── CON Opening Argument (quality score: {state['d2_quality_score']:.3f}) ───
{state['d2_argument']}

─── PRO Rebuttal (against CON) ───
{state['d1_rebuttal']}

─── CON Rebuttal (against PRO) ───
{state['d2_rebuttal']}

─── Automated Scoring (local NLI + semantic models) ───
  PRO — NLI entailment: {pro_nli:.3f}  |  Semantic alignment: {pro_sem:.3f}
  CON — NLI entailment: {con_nli:.3f}  |  Semantic alignment: {con_sem:.3f}

Your tasks:
1. Summarise the key clash points between the two sides (3-5 sentences).
2. Evaluate which side presented stronger evidence and logic overall,
   taking into account the automated scores above.
3. Declare a winner: "PRO", "CON", or "DRAW" with a brief justification.

Respond ONLY with valid JSON, no explanation:
{{
  "convergence_summary": "...",
  "winner": "PRO" | "CON" | "DRAW",
  "winner_justification": "..."
}}"""

    parsed        = _extract_json(_llama([{"role": "user", "content": prompt}], max_tokens=500))
    conv_summary  = parsed["convergence_summary"]
    winner        = parsed["winner"].upper()
    justification = parsed.get("winner_justification", "")
    full_summary  = f"{conv_summary}\n\n🏆 Winner: {winner} — {justification}"

    print(f"\n   📋 Summary       : {conv_summary}")
    print(f"   🏆 Winner        : {winner}")
    print(f"   📝 Justification : {justification}")
    print(f"   📊 PRO scores    : NLI={pro_nli:.3f}  SEM={pro_sem:.3f}")
    print(f"   📊 CON scores    : NLI={con_nli:.3f}  SEM={con_sem:.3f}")

    return {
        "convergence_summary": full_summary,
        "winner":              winner,
        "pro_nli_score":       pro_nli,
        "con_nli_score":       con_nli,
        "pro_semantic_score":  pro_sem,
        "con_semantic_score":  con_sem,
        "messages": [{"role": "convergence", "content": full_summary}],
    }


# =============================================================================
# Node 7 – Verdict  (Llama-3 via HF API, informed by local-model scores)
# =============================================================================

def verdict_node(state: OrchestratorState) -> dict:
    """
    Final node — synthesises all debate content AND all evidence into:
      1. A binary YES / NO / INCONCLUSIVE verdict.
      2. A one-sentence brief answer.
      3. A multi-paragraph elaborate explanation grounded in PubMed evidence.
    """
    print(f"\n{'='*70}")
    print("🏛️  VERDICT NODE – Generating final YES/NO answer")
    print(f"{'='*70}")

    def _fmt_arg_evidence(evidence_list: list, max_items: int = 5) -> str:
        lines = []
        for i, ev in enumerate(evidence_list[:max_items], 1):
            if isinstance(ev, dict):
                title   = ev.get("title", "")
                abstract = ev.get("abstract", ev.get("text", ""))
                snippet  = (abstract[:180] + "…") if len(abstract) > 180 else abstract
                lines.append(f"  {i}. [{title}] {snippet}")
            else:
                lines.append(f"  {i}. {str(ev)[:180]}")
        return "\n".join(lines) if lines else "  (none retrieved)"

    def _fmt_rebuttal_evidence(evidence_dict: dict) -> str:
        if not evidence_dict:
            return "  (none)"
        return "\n".join(
            f"  • \"{cp}\": {cnt} PubMed abstract(s)"
            for cp, cnt in evidence_dict.items()
        )

    pro_ev     = _fmt_arg_evidence(state.get("d1_evidence", []))
    con_ev     = _fmt_arg_evidence(state.get("d2_evidence", []))
    pro_reb_ev = _fmt_rebuttal_evidence(state.get("d1_rebuttal_evidence", {}))
    con_reb_ev = _fmt_rebuttal_evidence(state.get("d2_rebuttal_evidence", {}))

    pro_nli = state.get("pro_nli_score", 0.0)
    con_nli = state.get("con_nli_score", 0.0)
    pro_sem = state.get("pro_semantic_score", 0.0)
    con_sem = state.get("con_semantic_score", 0.0)

    prompt = f"""You are an expert biomedical analyst who has observed a full structured debate.
Your task: deliver a definitive, evidence-grounded verdict on the original question.

════════════════════════════════════════════════════════
ORIGINAL QUESTION:
{state['topic']}

FRAMED MOTION:
{state['framed_topic']}

BACKGROUND:
{state['context_summary']}
════════════════════════════════════════════════════════

── PRO ARGUMENT (quality {state['d1_quality_score']:.3f} / {state.get('d1_quality_level', '')}) ──
{state['d1_argument']}

  PubMed evidence used (top items):
{pro_ev}

── CON ARGUMENT (quality {state['d2_quality_score']:.3f} / {state.get('d2_quality_level', '')}) ──
{state['d2_argument']}

  PubMed evidence used (top items):
{con_ev}

── PRO REBUTTAL ──
  Logical flaws found in CON   : {state.get('d1_rebuttal_logical_flaws', [])}
  Counter-points raised        : {state.get('d1_rebuttal_counter_points', [])}
  PubMed evidence per counter-point:
{pro_reb_ev}
{state['d1_rebuttal']}

── CON REBUTTAL ──
  Logical flaws found in PRO   : {state.get('d2_rebuttal_logical_flaws', [])}
  Counter-points raised        : {state.get('d2_rebuttal_counter_points', [])}
  PubMed evidence per counter-point:
{con_reb_ev}
{state['d2_rebuttal']}

── AUTOMATED LOCAL-MODEL SCORES ──
  (bart-large-mnli NLI  +  all-MiniLM-L6-v2 semantic — both from local .model_store)
  PRO — NLI entailment: {pro_nli:.3f}  |  Semantic alignment: {pro_sem:.3f}
  CON — NLI entailment: {con_nli:.3f}  |  Semantic alignment: {con_sem:.3f}

── CONVERGENCE SUMMARY ──
{state['convergence_summary']}

── DEBATE WINNER ──
{state['winner']}
════════════════════════════════════════════════════════

Based on ALL of the above — arguments, PubMed evidence, quality scores,
rebuttal evidence, automated NLI/semantic scores, and the convergence —
answer the original question.

Return ONLY valid JSON (no preamble, no markdown fences):
{{
  "verdict_answer": "YES" | "NO" | "INCONCLUSIVE",
  "verdict_brief": "<one direct sentence answering the original question>",
  "verdict_elaborate": "<200-350 word multi-paragraph explanation that: (1) states the answer plainly, (2) highlights the strongest PRO PubMed evidence, (3) acknowledges the strongest CON evidence, (4) explains how the rebuttal evidence affected the balance, (5) references the NLI/semantic scores where relevant, (6) states why the verdict tips this way, (7) notes caveats or conditions>"
}}
Return STRICTLY valid JSON.
- No markdown, no extra text
- Escape all quotes properly
- Do not include newlines inside JSON strings unless escaped (\\n)
- Output must be parsable by Python json.loads()"""

    parsed            = _extract_json(_llama([{"role": "user", "content": prompt}], max_tokens=800))
    verdict_answer    = parsed["verdict_answer"].upper()
    verdict_brief     = parsed["verdict_brief"]
    verdict_elaborate = parsed["verdict_elaborate"]

    badge = {"YES": "✅ YES", "NO": "❌ NO", "INCONCLUSIVE": "❓ INCONCLUSIVE"}.get(
        verdict_answer, verdict_answer
    )
    print(f"\n   🏛️  Verdict Answer  : {badge}")
    print(f"   📝 Brief Answer    : {verdict_brief}")
    print(f"\n   📖 Elaborate Answer:\n")
    for para in verdict_elaborate.split("\n"):
        if para.strip():
            print(f"   {para.strip()}")

    return {
        "verdict_answer":    verdict_answer,
        "verdict_brief":     verdict_brief,
        "verdict_elaborate": verdict_elaborate,
        "messages": [{
            "role":    "verdict",
            "content": (
                f"Verdict: {verdict_answer}\n\n"
                f"{verdict_brief}\n\n"
                f"{verdict_elaborate}"
            ),
        }],
    }


# =============================================================================
# Graph Assembly
# =============================================================================

def create_orchestrator_graph():
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("moderator",         moderator_node)
    workflow.add_node("debater1_arg",      debater1_arg_node)
    workflow.add_node("debater2_arg",      debater2_arg_node)
    workflow.add_node("debater1_rebuttal", debater1_rebuttal_node)
    workflow.add_node("debater2_rebuttal", debater2_rebuttal_node)
    workflow.add_node("convergence",       convergence_node)
    workflow.add_node("verdict",           verdict_node)

    workflow.add_edge("moderator",         "debater1_arg")
    workflow.add_edge("debater1_arg",      "debater2_arg")
    workflow.add_edge("debater2_arg",      "debater1_rebuttal")
    workflow.add_edge("debater1_rebuttal", "debater2_rebuttal")
    workflow.add_edge("debater2_rebuttal", "convergence")
    workflow.add_edge("convergence",       "verdict")
    workflow.add_edge("verdict",           END)

    workflow.set_entry_point("moderator")
    return workflow.compile()


# =============================================================================
# Public Interface
# =============================================================================

def run_debate(
    topic: str,
    max_arg_iterations: int = 5,
    hf_api_key: str | None = None,
) -> dict:
    if hf_api_key:
        os.environ["HF_TOKEN"] = hf_api_key

    # Warm up local models once before the graph runs so the first node
    # doesn't pay the cold-start cost mid-pipeline.
    initialize_runtime_models()

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
        "  moderator → debater1_arg (PRO) → debater2_arg (CON)\n"
        "           → debater1_rebuttal → debater2_rebuttal\n"
        "           → convergence → verdict → END\n"
        "\nModels:\n"
        "  LLM text generation : meta-llama/Meta-Llama-3-8B-Instruct  "
        "(HuggingFace Inference API)\n"
        f"  NLI scoring         : {_NLI_MODEL_ID}  (local .model_store)\n"
        f"  Semantic scoring    : {_EMBED_MODEL_ID}  (local .model_store)\n"
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

        # Local-model scores surfaced at the top level for easy inspection
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

        # Convenience flat bundles
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

    # ── Final console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✅ DEBATE COMPLETE")
    print("=" * 80)
    print(f"Motion             : {result['framed_topic']}")

    print(f"\n🔵 PRO")
    print(f"   Quality   : {result['pro']['quality_score']:.3f}  "
          f"[{result['pro']['quality_level']}]  accepted={result['pro']['accepted']}")
    print(f"   Scores    : {result['pro']['scores']}")
    print(f"   Local NLI : {result['local_scores']['pro_nli']:.3f}  "
          f"Semantic: {result['local_scores']['pro_semantic']:.3f}")
    print(f"   Evidence  : {len(result['pro']['evidence'])} reranked piece(s)")
    print(f"   PubMed Q  : {result['pro']['pubmed_query']}")

    print(f"\n🔴 CON")
    print(f"   Quality   : {result['con']['quality_score']:.3f}  "
          f"[{result['con']['quality_level']}]  accepted={result['con']['accepted']}")
    print(f"   Scores    : {result['con']['scores']}")
    print(f"   Local NLI : {result['local_scores']['con_nli']:.3f}  "
          f"Semantic: {result['local_scores']['con_semantic']:.3f}")
    print(f"   Evidence  : {len(result['con']['evidence'])} reranked piece(s)")
    print(f"   PubMed Q  : {result['con']['pubmed_query']}")

    print(f"\n🔵 PRO Rebuttal")
    print(f"   Logical flaws   : {result['pro_rebuttal']['logical_flaws']}")
    print(f"   Counter-points  : {result['pro_rebuttal']['counter_points']}")
    print(f"   Evidence counts : {result['pro_rebuttal']['evidence_counts']}")

    print(f"\n🔴 CON Rebuttal")
    print(f"   Logical flaws   : {result['con_rebuttal']['logical_flaws']}")
    print(f"   Counter-points  : {result['con_rebuttal']['counter_points']}")
    print(f"   Evidence counts : {result['con_rebuttal']['evidence_counts']}")

    print(f"\n⚖️  Winner          : {result['winner']}")
    print(f"📋 Convergence:\n{result['convergence_summary']}")
    print(f"\n{'='*80}")
    print(f"🏛️  FINAL VERDICT   : {result['verdict']['answer']}")
    print(f"📝 Brief Answer    : {result['verdict']['brief']}")
    print(f"\n📖 Elaborate Answer:\n{result['verdict']['elaborate']}")
    print("=" * 80)

    return result


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_debate(
        topic=(
            "Are group 2 innate lymphoid cells (ILC2s) increased in "
            "chronic rhinosinusitis with nasal polyps or eosinophilia?"
        ),
        max_arg_iterations=5,
        hf_api_key=os.getenv("HF_TOKEN"),
    )