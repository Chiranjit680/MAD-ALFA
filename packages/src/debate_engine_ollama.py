"""
Debate Orchestrator  —  LangGraph
==================================
Pipeline:
    START
      │
      ▼
    moderator          → frames topic, writes context summary
      │
      ▼
    debater1_arg  (PRO)  → debate_argument_generator subgraph
      │
      ▼
    debater2_arg  (CON)  → debate_argument_generator subgraph
      │
      ▼
    debater1_rebuttal    → rebuttal subgraph  (PRO rebuts CON)
      │
      ▼
    debater2_rebuttal    → rebuttal subgraph  (CON rebuts PRO)
      │
      ▼
    convergence          → NLI + semantic scoring, winner declared
      │
      ▼
    verdict              → YES / NO / INCONCLUSIVE + elaboration
      │
      ▼
    END

LLM backend  : Ollama  (llama3, configurable)
NLI model    : facebook/bart-large-mnli        (local .model_store)
Embed model  : sentence-transformers/all-MiniLM-L6-v2  (local .model_store)
"""

from __future__ import annotations

import json
import operator
import os
import re
from typing import Annotated, TypedDict

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from transformers import pipeline as hf_pipeline

try:
    from .local_model_store import get_local_model_dir
except ImportError:
    from local_model_store import get_local_model_dir

from debate_agent import DebateState, create_debate_argument_graph
from model_inference import load_models
from rebuttal_node import RebuttalState, create_rebuttal_graph

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11436")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",    "llama3")
OLLAMA_TEMP      = float(os.getenv("OLLAMA_TEMP", "0.7"))

NLI_MODEL_ID     = "facebook/bart-large-mnli"
EMBED_MODEL_ID   = "sentence-transformers/all-MiniLM-L6-v2"

_HAS_CUDA  = torch.cuda.is_available()
_DEVICE_ID = 0 if _HAS_CUDA else -1
_DTYPE     = torch.float16 if _HAS_CUDA else torch.float32

# ---------------------------------------------------------------------------
# Lazy local-model handles
# ---------------------------------------------------------------------------

_nli_pipeline:   hf_pipeline | None = None
_embed_pipeline: hf_pipeline | None = None


# ---------------------------------------------------------------------------
# LLM helper  —  Ollama
# ---------------------------------------------------------------------------

def _llm(messages: list[dict], max_tokens: int = 600) -> str:
    """Send a list of {role, content} dicts to Ollama and return the reply."""
    lc_messages = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMP,
        num_predict=max_tokens,
    )
    for attempt in range(3):
        reply = llm.invoke(lc_messages)
        content = getattr(reply, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        print(f"   ⚠  Ollama returned an empty response (attempt {attempt + 1}/3)")
    raise RuntimeError("Ollama returned empty content after 3 attempts.")


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON; falls back to brace extraction."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


# ---------------------------------------------------------------------------
# Local model initialisation
# ---------------------------------------------------------------------------

def _load_pipeline(task: str, model_path: str, **kwargs) -> hf_pipeline:
    """Load a transformers pipeline with automatic CPU fallback on OOM."""
    try:
        return hf_pipeline(task, model=model_path, device=_DEVICE_ID,
                           torch_dtype=_DTYPE, **kwargs)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and _DEVICE_ID != -1:
            print("   ⚠  OOM — retrying on CPU")
            torch.cuda.empty_cache()
            return hf_pipeline(task, model=model_path, device=-1,
                               torch_dtype=torch.float32, **kwargs)
        raise


def init_local_models(force: bool = False) -> None:
    """Lazily load NLI and embedding models. Safe to call multiple times."""
    global _nli_pipeline, _embed_pipeline

    if not force and _nli_pipeline and _embed_pipeline:
        return

    device_label = f"GPU:{_DEVICE_ID}" if _HAS_CUDA else "CPU"
    print(f"\n── Loading local models ({device_label}, {_DTYPE}) ──")

    if force or _nli_pipeline is None:
        print(f"   NLI   : {NLI_MODEL_ID}")
        _nli_pipeline = _load_pipeline(
            "zero-shot-classification",
            get_local_model_dir(NLI_MODEL_ID),
        )
        print("   ✓ NLI ready")

    if force or _embed_pipeline is None:
        print(f"   Embed : {EMBED_MODEL_ID}")
        _embed_pipeline = _load_pipeline(
            "feature-extraction",
            get_local_model_dir(EMBED_MODEL_ID),
            return_tensor=False,
        )
        print("   ✓ Embedding ready")

    print("── Local models ready ──\n")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _nli_score(premise: str, hypothesis: str) -> float:
    """Return bart-large-mnli entailment probability (0-1)."""
    if _nli_pipeline is None:
        init_local_models()
    try:
        result = _nli_pipeline(
            premise[:1024],
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="{}",
            multi_label=False,
        )
        return float(dict(zip(result["labels"], result["scores"])).get("entailment", 0.0))
    except Exception as exc:
        print(f"   ⚠  NLI error: {exc}")
        return 0.0


def _embed(text: str) -> np.ndarray:
    """Mean-pool token embeddings into a single vector."""
    if _embed_pipeline is None:
        init_local_models()
    raw = np.array(_embed_pipeline(text[:512]))
    return raw[0].mean(axis=0) if raw.ndim == 3 else raw.mean(axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _semantic_score(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using all-MiniLM-L6-v2."""
    try:
        return _cosine(_embed(text_a), _embed(text_b))
    except Exception as exc:
        print(f"   ⚠  Semantic error: {exc}")
        return 0.0


def _score_arguments(framed_topic: str, pro_arg: str, con_arg: str) -> dict:
    """Run NLI + semantic scoring for both arguments."""
    print("   🔬 Scoring arguments …")
    scores = {
        "pro_nli":      _nli_score(pro_arg, framed_topic),
        "con_nli":      _nli_score(con_arg, framed_topic),
        "pro_semantic": _semantic_score(pro_arg, framed_topic),
        "con_semantic": _semantic_score(con_arg, framed_topic),
    }
    print(f"   NLI      PRO={scores['pro_nli']:.3f}  CON={scores['con_nli']:.3f}")
    print(f"   Semantic PRO={scores['pro_semantic']:.3f}  CON={scores['con_semantic']:.3f}")
    return scores


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    # inputs
    topic:               str
    max_arg_iterations:  int

    # moderator
    framed_topic:        str
    context_summary:     str

    # PRO argument
    d1_argument:         str
    d1_quality_score:    float
    d1_quality_level:    str
    d1_pubmed_query:     str
    d1_iterations_used:  int
    d1_retrieval_attempts: int
    d1_accepted:         bool
    d1_scores:           dict
    d1_evidence:         list

    # CON argument
    d2_argument:         str
    d2_quality_score:    float
    d2_quality_level:    str
    d2_pubmed_query:     str
    d2_iterations_used:  int
    d2_retrieval_attempts: int
    d2_accepted:         bool
    d2_scores:           dict
    d2_evidence:         list

    # PRO rebuttal
    d1_rebuttal:                str
    d1_rebuttal_logical_flaws:  list
    d1_rebuttal_counter_points: list
    d1_rebuttal_evidence:       dict

    # CON rebuttal
    d2_rebuttal:                str
    d2_rebuttal_logical_flaws:  list
    d2_rebuttal_counter_points: list
    d2_rebuttal_evidence:       dict

    # convergence
    convergence_summary: str
    winner:              str   # "PRO" | "CON" | "DRAW"

    # local-model scores
    pro_nli_score:       float
    con_nli_score:       float
    pro_semantic_score:  float
    con_semantic_score:  float

    # verdict
    verdict_answer:      str   # "YES" | "NO" | "INCONCLUSIVE"
    verdict_brief:       str
    verdict_elaborate:   str

    # trace
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Helpers shared across nodes
# ---------------------------------------------------------------------------

_BLANK_DEBATE_STATE: DebateState = {
    "topic": "", "stance": "", "pubmed_query": "", "argument": "",
    "iteration": 1, "max_iterations": 5, "pubmed_abstracts": [],
    "reranked_evidence": [], "nli_score": 0.0, "semantic_score": 0.0,
    "logprob_score": 0.0, "confidence_score": 0.0, "quality_score": 0.0,
    "quality_level": "", "accepted": False, "feedback": "", "prev_arg": "",
    "opposition_arg": "", "messages": [], "retrieval_feedback": "",
    "retrieval_attempts": 0, "failure_reason": "",
}


def _run_debate_subgraph(
    topic: str,
    stance: str,
    max_iterations: int,
    opposition_arg: str = "",
) -> DebateState:
    init: DebateState = {
        **_BLANK_DEBATE_STATE,
        "topic":         topic,
        "stance":        stance,
        "max_iterations": max_iterations,
        "opposition_arg": opposition_arg,
    }
    return create_debate_argument_graph().invoke(init)


def _run_rebuttal_subgraph(original_argument: str, topic: str) -> RebuttalState:
    init: RebuttalState = {
        "original_argument": original_argument,
        "topic":             topic,
        "logical_flaws":     [],
        "counter_points":    [],
        "pubmed_evidence":   {},
        "rebuttal":          "",
        "messages":          [],
    }
    return create_rebuttal_graph().invoke(init)


def _fmt_evidence(evidence_list: list, max_items: int = 5) -> str:
    lines = []
    for i, ev in enumerate(evidence_list[:max_items], 1):
        if isinstance(ev, dict):
            title    = ev.get("title", "")
            abstract = ev.get("abstract", ev.get("text", ""))
            snippet  = (abstract[:180] + "…") if len(abstract) > 180 else abstract
            lines.append(f"  {i}. [{title}] {snippet}")
        else:
            lines.append(f"  {i}. {str(ev)[:180]}")
    return "\n".join(lines) or "  (none retrieved)"


def _fmt_rebuttal_evidence(evidence_dict: dict) -> str:
    if not evidence_dict:
        return "  (none)"
    return "\n".join(
        f"  • \"{cp}\": {cnt} abstract(s)"
        for cp, cnt in evidence_dict.items()
    )


# ---------------------------------------------------------------------------
# Node 1 — Moderator
# ---------------------------------------------------------------------------

def moderator_node(state: OrchestratorState) -> dict:
    print("\n══ MODERATOR ══")

    prompt = f"""You are a professional debate moderator.

Given the debate topic below, produce:
1. A clear, neutral one-sentence debate motion.
2. A 2-3 sentence background summary that contextualises the topic without taking sides.

Topic: {state["topic"]}

Respond with ONLY valid JSON:
{{
  "framed_topic": "...",
  "context_summary": "..."
}}"""

    parsed = _parse_json(_llm([{"role": "user", "content": prompt}], max_tokens=300))
    print(f"   Motion  : {parsed['framed_topic']}")
    print(f"   Context : {parsed['context_summary']}")

    return {
        "framed_topic":    parsed["framed_topic"],
        "context_summary": parsed["context_summary"],
        "messages": [{"role": "moderator",
                      "content": f"Motion: {parsed['framed_topic']}"}],
    }


# ---------------------------------------------------------------------------
# Node 2 — Debater 1 (PRO) argument
# ---------------------------------------------------------------------------

def debater1_arg_node(state: OrchestratorState) -> dict:
    print("\n══ PRO — opening argument ══")
    final = _run_debate_subgraph(
        topic=state["framed_topic"],
        stance="PRO",
        max_iterations=state.get("max_arg_iterations", 5),
    )
    print(f"   quality={final['quality_score']:.3f}  evidence={len(final['reranked_evidence'])}")
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


# ---------------------------------------------------------------------------
# Node 3 — Debater 2 (CON) argument
# ---------------------------------------------------------------------------

def debater2_arg_node(state: OrchestratorState) -> dict:
    print("\n══ CON — opening argument ══")
    final = _run_debate_subgraph(
        topic=state["framed_topic"],
        stance="CON",
        max_iterations=state.get("max_arg_iterations", 5),
        opposition_arg=state["d1_argument"],
    )
    print(f"   quality={final['quality_score']:.3f}  evidence={len(final['reranked_evidence'])}")
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


# ---------------------------------------------------------------------------
# Node 4 — Debater 1 (PRO) rebuttal
# ---------------------------------------------------------------------------

def debater1_rebuttal_node(state: OrchestratorState) -> dict:
    print("\n══ PRO — rebuttal ══")
    final    = _run_rebuttal_subgraph(state["d2_argument"], state["framed_topic"])
    ev_counts = {cp: len(abs_) for cp, abs_ in final["pubmed_evidence"].items()}
    print(f"   flaws={len(final['logical_flaws'])}  "
          f"counter-points={len(final['counter_points'])}  "
          f"abstracts={sum(ev_counts.values())}")
    return {
        "d1_rebuttal":                final["rebuttal"],
        "d1_rebuttal_logical_flaws":  final["logical_flaws"],
        "d1_rebuttal_counter_points": final["counter_points"],
        "d1_rebuttal_evidence":       ev_counts,
        "messages": [{"role": "debater1_rebuttal", "content": final["rebuttal"]}],
    }


# ---------------------------------------------------------------------------
# Node 5 — Debater 2 (CON) rebuttal
# ---------------------------------------------------------------------------

def debater2_rebuttal_node(state: OrchestratorState) -> dict:
    print("\n══ CON — rebuttal ══")
    final    = _run_rebuttal_subgraph(state["d1_argument"], state["framed_topic"])
    ev_counts = {cp: len(abs_) for cp, abs_ in final["pubmed_evidence"].items()}
    print(f"   flaws={len(final['logical_flaws'])}  "
          f"counter-points={len(final['counter_points'])}  "
          f"abstracts={sum(ev_counts.values())}")
    return {
        "d2_rebuttal":                final["rebuttal"],
        "d2_rebuttal_logical_flaws":  final["logical_flaws"],
        "d2_rebuttal_counter_points": final["counter_points"],
        "d2_rebuttal_evidence":       ev_counts,
        "messages": [{"role": "debater2_rebuttal", "content": final["rebuttal"]}],
    }


# ---------------------------------------------------------------------------
# Node 6 — Convergence
# ---------------------------------------------------------------------------

def convergence_node(state: OrchestratorState) -> dict:
    print("\n══ CONVERGENCE ══")

    scores  = _score_arguments(
        state["framed_topic"], state["d1_argument"], state["d2_argument"]
    )

    prompt = f"""You are a professional debate moderator closing the debate.

Motion: {state['framed_topic']}

── PRO opening argument (quality {state['d1_quality_score']:.3f}) ──
{state['d1_argument']}

── CON opening argument (quality {state['d2_quality_score']:.3f}) ──
{state['d2_argument']}

── PRO rebuttal ──
{state['d1_rebuttal']}

── CON rebuttal ──
{state['d2_rebuttal']}

── Automated scores (NLI + semantic) ──
  PRO  NLI={scores['pro_nli']:.3f}  Semantic={scores['pro_semantic']:.3f}
  CON  NLI={scores['con_nli']:.3f}  Semantic={scores['con_semantic']:.3f}

Tasks:
1. Summarise the key clash points (3-5 sentences).
2. Evaluate which side presented stronger evidence and logic.
3. Declare a winner.

Respond with ONLY valid JSON:
{{
  "convergence_summary": "...",
  "winner": "PRO" | "CON" | "DRAW",
  "winner_justification": "..."
}}"""

    parsed = _parse_json(_llm([{"role": "user", "content": prompt}], max_tokens=500))
    winner  = parsed["winner"].upper()
    summary = (
        f"{parsed['convergence_summary']}\n\n"
        f"🏆 Winner: {winner} — {parsed.get('winner_justification', '')}"
    )

    print(f"   Winner: {winner}")

    return {
        "convergence_summary": summary,
        "winner":              winner,
        "pro_nli_score":       scores["pro_nli"],
        "con_nli_score":       scores["con_nli"],
        "pro_semantic_score":  scores["pro_semantic"],
        "con_semantic_score":  scores["con_semantic"],
        "messages": [{"role": "convergence", "content": summary}],
    }


# ---------------------------------------------------------------------------
# Node 7 — Verdict
# ---------------------------------------------------------------------------

def verdict_node(state: OrchestratorState) -> dict:
    print("\n══ VERDICT ══")

    prompt = f"""You are an expert biomedical analyst who has observed a full structured debate.
Deliver a definitive, evidence-grounded verdict on the original question.

════════════════════════════════════
ORIGINAL QUESTION:
{state['topic']}

FRAMED MOTION:
{state['framed_topic']}

BACKGROUND:
{state['context_summary']}
════════════════════════════════════

── PRO argument (quality {state['d1_quality_score']:.3f} / {state.get('d1_quality_level', '')}) ──
{state['d1_argument']}

  Top PubMed evidence:
{_fmt_evidence(state.get('d1_evidence', []))}

── CON argument (quality {state['d2_quality_score']:.3f} / {state.get('d2_quality_level', '')}) ──
{state['d2_argument']}

  Top PubMed evidence:
{_fmt_evidence(state.get('d2_evidence', []))}

── PRO rebuttal ──
  Logical flaws found : {state.get('d1_rebuttal_logical_flaws', [])}
  Counter-points      : {state.get('d1_rebuttal_counter_points', [])}
  PubMed evidence     :
{_fmt_rebuttal_evidence(state.get('d1_rebuttal_evidence', {}))}
{state['d1_rebuttal']}

── CON rebuttal ──
  Logical flaws found : {state.get('d2_rebuttal_logical_flaws', [])}
  Counter-points      : {state.get('d2_rebuttal_counter_points', [])}
  PubMed evidence     :
{_fmt_rebuttal_evidence(state.get('d2_rebuttal_evidence', {}))}
{state['d2_rebuttal']}

── Automated scores ──
  PRO  NLI={state.get('pro_nli_score', 0):.3f}  Semantic={state.get('pro_semantic_score', 0):.3f}
  CON  NLI={state.get('con_nli_score', 0):.3f}  Semantic={state.get('con_semantic_score', 0):.3f}

── Convergence summary ──
{state['convergence_summary']}

── Debate winner ──
{state['winner']}
════════════════════════════════════

Based on ALL of the above, answer the original question.

Return ONLY valid JSON (no preamble, no markdown):
{{
  "verdict_answer":    "YES" | "NO" ,
  "verdict_brief":     "<one direct sentence>",
  "verdict_elaborate": "<200-350 word multi-paragraph explanation: (1) plain answer, (2) strongest PRO evidence, (3) strongest CON evidence, (4) how rebuttals shifted the balance, (5) NLI/semantic score commentary, (6) why verdict tips this way, (7) caveats>"
}}"""

    try:
        parsed = _parse_json(_llm([{"role": "user", "content": prompt}], max_tokens=800))
        answer = str(parsed["verdict_answer"]).upper()
        brief = str(parsed["verdict_brief"])
        elaborate = str(parsed["verdict_elaborate"])
    except Exception as exc:
        print(f"   ⚠  Verdict JSON parse failed: {exc}")
        winner = str(state.get("winner", "DRAW")).upper()
        if winner == "PRO":
            answer = "YES"
        elif winner == "CON":
            answer = "NO"
        else:
            answer = "INCONCLUSIVE"

        brief = (
            "Evidence is mixed; the available debate output does not support a single "
            "high-confidence binary conclusion."
            if answer == "INCONCLUSIVE"
            else f"Based on the debate synthesis, the best-supported answer is {answer}."
        )
        elaborate = (
            "The final verdict was produced via fallback logic because the LLM response "
            "could not be parsed as valid JSON in the verdict node. The fallback uses the "
            "convergence winner and local scoring context to avoid terminating the pipeline. "
            "Review the logs and rerun with a smaller prompt context or lower generation "
            "length if you need a fully model-authored verdict narrative."
        )

    badge = {"YES": "✅ YES", "NO": "❌ NO"}.get(answer, answer)
    print(f"   {badge}")
    print(f"   {brief}")

    return {
        "verdict_answer":    answer,
        "verdict_brief":     brief,
        "verdict_elaborate": elaborate,
        "messages": [{
            "role":    "verdict",
            "content": f"Verdict: {answer}\n\n{brief}\n\n{elaborate}",
        }],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    wf = StateGraph(OrchestratorState)

    wf.add_node("moderator",          moderator_node)
    wf.add_node("debater1_arg",       debater1_arg_node)
    wf.add_node("debater2_arg",       debater2_arg_node)
    wf.add_node("debater1_rebuttal",  debater1_rebuttal_node)
    wf.add_node("debater2_rebuttal",  debater2_rebuttal_node)
    wf.add_node("convergence",        convergence_node)
    wf.add_node("verdict",            verdict_node)

    wf.set_entry_point("moderator")
    wf.add_edge("moderator",         "debater1_arg")
    wf.add_edge("debater1_arg",      "debater2_arg")
    wf.add_edge("debater2_arg",      "debater1_rebuttal")
    wf.add_edge("debater1_rebuttal", "debater2_rebuttal")
    wf.add_edge("debater2_rebuttal", "convergence")
    wf.add_edge("convergence",       "verdict")
    wf.add_edge("verdict",           END)

    return wf.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_debate(topic: str, max_arg_iterations: int = 5) -> dict:
    """Run a full structured debate and return the result dict."""
    init_local_models()

    initial: OrchestratorState = {
        "topic":               topic,
        "max_arg_iterations":  max_arg_iterations,
        "framed_topic":        "",
        "context_summary":     "",
        "d1_argument":         "", "d1_quality_score":    0.0,
        "d1_quality_level":    "", "d1_pubmed_query":     "",
        "d1_iterations_used":  0,  "d1_retrieval_attempts": 0,
        "d1_accepted":         False, "d1_scores":          {}, "d1_evidence": [],
        "d2_argument":         "", "d2_quality_score":    0.0,
        "d2_quality_level":    "", "d2_pubmed_query":     "",
        "d2_iterations_used":  0,  "d2_retrieval_attempts": 0,
        "d2_accepted":         False, "d2_scores":          {}, "d2_evidence": [],
        "d1_rebuttal":                "", "d1_rebuttal_logical_flaws":  [],
        "d1_rebuttal_counter_points": [], "d1_rebuttal_evidence":       {},
        "d2_rebuttal":                "", "d2_rebuttal_logical_flaws":  [],
        "d2_rebuttal_counter_points": [], "d2_rebuttal_evidence":       {},
        "convergence_summary": "", "winner":              "",
        "pro_nli_score":       0.0, "con_nli_score":      0.0,
        "pro_semantic_score":  0.0, "con_semantic_score": 0.0,
        "verdict_answer":      "", "verdict_brief":       "",
        "verdict_elaborate":   "", "messages":            [],
    }

    print(f"\n{'═'*70}")
    print(f"  DEBATE ORCHESTRATOR")
    print(f"  Topic  : {topic}")
    print(f"  Model  : {OLLAMA_MODEL}  @  {OLLAMA_BASE_URL}")
    print(f"  Iters  : {max_arg_iterations}")
    print(f"{'═'*70}\n")

    fs = build_graph().invoke(initial)

    result = {
        "topic":           fs["topic"],
        "framed_topic":    fs["framed_topic"],
        "context_summary": fs["context_summary"],

        "pro": {
            "argument":           fs["d1_argument"],
            "quality_score":      fs["d1_quality_score"],
            "quality_level":      fs["d1_quality_level"],
            "pubmed_query":       fs["d1_pubmed_query"],
            "iterations_used":    fs["d1_iterations_used"],
            "retrieval_attempts": fs["d1_retrieval_attempts"],
            "accepted":           fs["d1_accepted"],
            "scores":             fs["d1_scores"],
            "evidence":           fs["d1_evidence"],
        },

        "con": {
            "argument":           fs["d2_argument"],
            "quality_score":      fs["d2_quality_score"],
            "quality_level":      fs["d2_quality_level"],
            "pubmed_query":       fs["d2_pubmed_query"],
            "iterations_used":    fs["d2_iterations_used"],
            "retrieval_attempts": fs["d2_retrieval_attempts"],
            "accepted":           fs["d2_accepted"],
            "scores":             fs["d2_scores"],
            "evidence":           fs["d2_evidence"],
        },

        "pro_rebuttal": {
            "rebuttal":        fs["d1_rebuttal"],
            "logical_flaws":   fs["d1_rebuttal_logical_flaws"],
            "counter_points":  fs["d1_rebuttal_counter_points"],
            "evidence_counts": fs["d1_rebuttal_evidence"],
        },

        "con_rebuttal": {
            "rebuttal":        fs["d2_rebuttal"],
            "logical_flaws":   fs["d2_rebuttal_logical_flaws"],
            "counter_points":  fs["d2_rebuttal_counter_points"],
            "evidence_counts": fs["d2_rebuttal_evidence"],
        },

        "convergence_summary": fs["convergence_summary"],
        "winner":              fs["winner"],

        "local_scores": {
            "pro_nli":      fs["pro_nli_score"],
            "con_nli":      fs["con_nli_score"],
            "pro_semantic": fs["pro_semantic_score"],
            "con_semantic": fs["con_semantic_score"],
        },

        "verdict": {
            "answer":    fs["verdict_answer"],
            "brief":     fs["verdict_brief"],
            "elaborate": fs["verdict_elaborate"],
        },

        "messages": fs["messages"],

        # flat evidence bundles for downstream use
        "pro_evidence": fs["d1_evidence"] + [
            {"counter_point": k, "count": v}
            for k, v in fs["d1_rebuttal_evidence"].items()
        ],
        "con_evidence": fs["d2_evidence"] + [
            {"counter_point": k, "count": v}
            for k, v in fs["d2_rebuttal_evidence"].items()
        ],
    }

    # ── Final summary ────────────────────────────────────────────────────────
    ls = result["local_scores"]
    print(f"\n{'═'*70}")
    print("  DEBATE COMPLETE")
    print(f"{'═'*70}")
    print(f"  Motion   : {result['framed_topic']}")
    print(f"\n  PRO  quality={result['pro']['quality_score']:.3f}"
          f"  NLI={ls['pro_nli']:.3f}  sem={ls['pro_semantic']:.3f}"
          f"  evidence={len(result['pro']['evidence'])}")
    print(f"  CON  quality={result['con']['quality_score']:.3f}"
          f"  NLI={ls['con_nli']:.3f}  sem={ls['con_semantic']:.3f}"
          f"  evidence={len(result['con']['evidence'])}")
    print(f"\n  Winner   : {result['winner']}")
    print(f"\n  Verdict  : {result['verdict']['answer']}")
    print(f"  Brief    : {result['verdict']['brief']}")
    print(f"\n  Elaborate:\n  {result['verdict']['elaborate']}")
    print(f"{'═'*70}\n")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_models("argument_quality_model_4features.pth")

    result = run_debate(
        topic=(
            "Are group 2 innate lymphoid cells (ILC2s) increased in "
            "chronic rhinosinusitis with nasal polyps or eosinophilia?"
        ),
        max_arg_iterations=5,
    )

    print(result["verdict"]["answer"])