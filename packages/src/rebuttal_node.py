"""
Debate Rebuttal Generator
Architecture:
  analyze_node -> rebuttal_pubmed_node -> rebuttal_node

Refactor goals:
  - Run on both GPU and CPU (PC-safe fallback)
  - Load runtime models lazily (no heavy import-time initialization)
  - Use local model cache via local_model_store.get_local_model_dir
  - Use d4data/biomedical-ner-all for biomedical NER
  - Use local Mistral-7B via get_judge_lm.mistral_inference instead of HF API
"""

from typing import TypedDict, Annotated
import argparse
import operator
import os
import json
import re
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import torch
from langgraph.graph import StateGraph, END
from transformers import pipeline as hf_pipeline

try:
    from .local_model_store import get_local_model_dir
except ImportError:
    from local_model_store import get_local_model_dir

# Import local Mistral inference — replaces HuggingFace Inference API calls
try:
    from .get_judge_lm import llm_inference
except ImportError:
    from get_judge_lm import llm_inference

load_dotenv()  # Load environment variables from .env file if present

# ============================================================
# Runtime model configuration
# ============================================================

_HAS_CUDA = torch.cuda.is_available()
_HF_DEVICE_ID = int(os.getenv("REBUTTAL_GPU_DEVICE", "0")) if _HAS_CUDA else -1
_NER_MODEL_ID = "d4data/biomedical-ner-all"

_ner_pipeline = None


def initialize_runtime_models(force_reload: bool = False) -> None:
    """Lazily initialize runtime models with local cache and CPU fallback."""
    global _ner_pipeline

    if not force_reload and _ner_pipeline is not None:
        return

    print("\n🚀 Initializing rebuttal runtime models...")
    print(f"   CUDA available: {_HAS_CUDA}")
    print(f"   Device id     : {_HF_DEVICE_ID}")
    if _HAS_CUDA:
        print(f"   GPU name      : {torch.cuda.get_device_name(_HF_DEVICE_ID)}")

    ner_model_path = get_local_model_dir(_NER_MODEL_ID)
    pipeline_dtype = torch.float16 if _HAS_CUDA else torch.float32

    _ner_pipeline = hf_pipeline(
        "ner",
        model=ner_model_path,
        aggregation_strategy="simple",
        device=_HF_DEVICE_ID,
        torch_dtype=pipeline_dtype,
    )

    print(f"✅ NER model loaded from local cache: {ner_model_path}")


# ============================================================
# State Definition
# ============================================================

class RebuttalState(TypedDict):
    original_argument: str
    topic: str
    logical_flaws: list[str]
    counter_points: list[str]
    pubmed_evidence: dict[str, list[str]]
    rebuttal: str
    messages: Annotated[list, operator.add]


# ============================================================
# NER helpers
# ============================================================


def _rebuttal_ner_extract(text: str) -> list[str]:
    """Extract biomedical entities from text for retrieval expansion."""
    if _ner_pipeline is None:
        initialize_runtime_models()

    seen = set()
    entities = []

    try:
        for ent in _ner_pipeline(text):
            mention = ent["word"].strip()
            if (
                mention
                and mention.lower() not in seen
                and len(mention) > 2
                and not mention.startswith("##")
            ):
                seen.add(mention.lower())
                entities.append(mention)
    except Exception as exc:
        print(f"   ⚠️  NER error: {exc}")

    return entities


# ============================================================
# PubMed helpers
# ============================================================


def _rebuttal_fetch_by_term(term: str, retmax: int = 5) -> list[str]:
    try:
        pmids = (
            requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"},
                timeout=15,
            )
            .json()
            .get("esearchresult", {})
            .get("idlist", [])
        )
        if not pmids:
            return []

        root = ET.fromstring(
            requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
                timeout=15,
            ).content.decode("utf-8", errors="replace")
        )

        abstracts = []
        for article in root.findall(".//PubmedArticle"):
            text = " ".join(
                s.text.strip() for s in article.findall(".//AbstractText") if s.text
            )
            if text:
                abstracts.append(text)

        return abstracts

    except Exception as exc:
        print(f"   ⚠️  PubMed fetch failed for '{term}': {exc}")
        return []


def _fetch_evidence_for_point(counter_point: str, retmax: int = 8) -> list[str]:
    """Fetch PubMed evidence for one counter-point using full query + NER expansion."""
    all_abstracts = []

    print(f"\n   🔍 [Full query] {counter_point}")
    full_query_abstracts = _rebuttal_fetch_by_term(counter_point, retmax=retmax)
    print(f"      -> {len(full_query_abstracts)} abstract(s)")
    all_abstracts.extend(full_query_abstracts)

    entities = _rebuttal_ner_extract(counter_point)
    print(f"   🧬 NER extracted {len(entities)} entity/entities: {entities}")

    per_entity_retmax = max(3, retmax // max(len(entities), 1))
    for entity in entities:
        print(f"   🔍 [NER entity] {entity}")
        entity_abstracts = _rebuttal_fetch_by_term(entity, retmax=per_entity_retmax)
        print(f"      -> {len(entity_abstracts)} abstract(s)")
        all_abstracts.extend(entity_abstracts)

    seen = set()
    deduped = []
    for abstract in all_abstracts:
        key = abstract[:120].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(abstract)

    print(
        f"   📚 Total after dedup: {len(deduped)} abstracts "
        f"({len(all_abstracts)} raw -> {len(deduped)} unique)"
    )

    return deduped


# ============================================================
# LLM helper — now uses local Mistral-7B via get_judge_lm
# ============================================================


def _local_chat(messages: list, max_new_tokens: int = 800) -> str:
    """
    Replace the remote HuggingFace Inference API call with a local
    Mistral-7B inference call via get_judge_lm.llm_inference.

    The 'messages' list follows the OpenAI chat format:
      [{"role": "system"|"user"|"assistant", "content": "..."}]

    We flatten the list into a single prompt string that Mistral's
    instruction format understands:
      <s>[INST] {user_content} [/INST]
    System messages are prepended before the first [INST] block.
    """
    system_parts = []
    turn_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            turn_parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            # Assistant turns go between instruction blocks (multi-turn)
            turn_parts.append(content)

    # Build the final prompt in Mistral instruct format
    system_prefix = (" ".join(system_parts) + "\n\n") if system_parts else ""
    prompt = "<s>" + system_prefix + " ".join(turn_parts)

    print(f"   📡 Calling local Mistral-7B (max_new_tokens={max_new_tokens}) ...")
    response = llm_inference(prompt, max_new_tokens=max_new_tokens)
    print("   ✅ Response received.")
    return response.strip()


def _extract_json_block(raw_text: str) -> dict:
    """Robustly parse JSON object from model output."""
    cleaned = re.sub(r"```json|```", "", (raw_text or "")).strip()

    if not cleaned:
        raise ValueError("Model returned empty text; expected JSON object.")

    # Prefer explicit fenced JSON when present.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text or "", re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as inner:
                raise inner from exc
        raise


def _fallback_analysis(topic: str) -> dict:
    """Safe defaults when model output is not valid JSON."""
    return {
        "logical_flaws": [
            "Overgeneralization from limited or unspecified evidence.",
            "Causal claims are asserted without adequately ruling out confounders.",
            "Key assumptions are unstated and not empirically validated.",
        ],
        "counter_points": [
            f"The claim about {topic} is too broad and not consistently supported by high-quality evidence.",
            "Competing explanations can account for the observed outcomes, weakening the argument's causal certainty.",
            "Methodological limitations and potential bias reduce confidence in the argument's conclusions.",
        ],
    }


# ============================================================
# Node 1 - Analyze
# ============================================================


def analyze_node(state: RebuttalState) -> dict:
    print(f"\n{'='*60}")
    print("🔎 ANALYZE NODE - Finding flaws and counter-points")
    print(f"{'='*60}")

    prompt = f"""You are an expert debate analyst and logician.

You will be given an argument on a topic. Your job is to:
1. Identify logical flaws in the argument (fallacies, unsupported claims, weak evidence, etc.)
2. Generate up to 3 strong, distinct counter-points that directly attack the argument

Topic: {state['topic']}

Argument to analyze:
\"\"\"{state['original_argument']}\"\"\"

Respond with ONLY a valid JSON object in this exact format:
{{
  \"logical_flaws\": [
    \"flaw 1 description\",
    \"flaw 2 description\"
  ],
  \"counter_points\": [
    \"counter-point 1 as a short declarative claim (10-20 words)\",
    \"counter-point 2 as a short declarative claim (10-20 words)\",
    \"counter-point 3 as a short declarative claim (10-20 words)\"
  ]
}}

Rules:
- Maximum 3 counter-points
- Each counter-point must be a specific, searchable claim
- Logical flaws must name the specific fallacy or weakness
- Output ONLY the JSON, no explanation"""

    # CHANGED: was _llama_chat(...), now _local_chat(...)
    raw = _local_chat(messages=[{"role": "user", "content": prompt}], max_new_tokens=600)
    try:
        parsed = _extract_json_block(raw)
    except Exception as exc:
        print(f"   ⚠️  Could not parse JSON from analyzer output: {exc}")
        preview = (raw or "").strip().replace("\n", " ")[:220]
        if preview:
            print(f"   ↪️  Raw preview: {preview}")
        parsed = _fallback_analysis(state["topic"])

    logical_flaws = parsed.get("logical_flaws", []) if isinstance(parsed, dict) else []
    counter_points = parsed.get("counter_points", []) if isinstance(parsed, dict) else []

    # Normalize output schema and keep the graph progressing even on noisy model output.
    if not isinstance(logical_flaws, list):
        logical_flaws = [str(logical_flaws)]
    if not isinstance(counter_points, list):
        counter_points = [str(counter_points)]

    logical_flaws = [str(x).strip() for x in logical_flaws if str(x).strip()][:5]
    counter_points = [str(x).strip() for x in counter_points if str(x).strip()][:3]

    if not counter_points:
        fallback = _fallback_analysis(state["topic"])
        logical_flaws = logical_flaws or fallback["logical_flaws"]
        counter_points = fallback["counter_points"][:3]

    print(f"\n   🧠 Logical Flaws Found ({len(logical_flaws)}):")
    for idx, flaw in enumerate(logical_flaws, 1):
        print(f"      {idx}. {flaw}")

    print(f"\n   ⚔️  Counter-Points Generated ({len(counter_points)}):")
    for idx, cp in enumerate(counter_points, 1):
        print(f"      {idx}. {cp}")

    return {
        "logical_flaws": logical_flaws,
        "counter_points": counter_points,
        "messages": [{
            "role": "analyze",
            "content": f"Found {len(logical_flaws)} flaws, {len(counter_points)} counter-points",
        }],
    }


# ============================================================
# Node 2 - PubMed retrieval
# ============================================================


def rebuttal_pubmed_node(state: RebuttalState) -> dict:
    print(f"\n{'='*60}")
    print("🔬 PUBMED NODE - Fetching evidence per counter-point")
    print(f"{'='*60}")

    pubmed_evidence = {}
    counter_points = state["counter_points"]

    for idx, counter_point in enumerate(counter_points, 1):
        print(f"\n   [{idx}/{len(counter_points)}] Searching for: {counter_point}")
        abstracts = _fetch_evidence_for_point(counter_point, retmax=8)
        pubmed_evidence[counter_point] = abstracts
        print(f"      -> {len(abstracts)} abstract(s) retrieved")

    total_abstracts = sum(len(v) for v in pubmed_evidence.values())
    print(f"\n   📚 Total abstracts across all counter-points: {total_abstracts}")

    return {
        "pubmed_evidence": pubmed_evidence,
        "messages": [{
            "role": "pubmed",
            "content": (
                f"Retrieved evidence for {len(counter_points)} counter-points "
                f"({total_abstracts} total abstracts)"
            ),
        }],
    }


# ============================================================
# Node 3 - Rebuttal synthesis
# ============================================================


def rebuttal_node(state: RebuttalState) -> dict:
    print(f"\n{'='*60}")
    print("✍️  REBUTTAL NODE - Synthesizing final rebuttal")
    print(f"{'='*60}")

    evidence_sections = []
    for counter_point, abstracts in state["pubmed_evidence"].items():
        top_abstracts = abstracts[:3]
        evidence_text = "\n".join(
            f"  [Evidence {idx+1}]: {abstract[:300]}..."
            for idx, abstract in enumerate(top_abstracts)
        ) or "  No direct PubMed evidence found - rely on logical reasoning."

        evidence_sections.append(f"Counter-Point: {counter_point}\n{evidence_text}")

    evidence_block = "\n\n".join(evidence_sections)
    flaws_block = "\n".join(f"- {flaw}" for flaw in state["logical_flaws"])
    cp_block = "\n".join(f"{idx+1}. {cp}" for idx, cp in enumerate(state["counter_points"]))

    prompt = f"""You are an expert debate rebuttal writer.

Topic: {state['topic']}

ORIGINAL ARGUMENT (to rebut):
\"\"\"{state['original_argument']}\"\"\"

LOGICAL FLAWS IDENTIFIED:
{flaws_block}

COUNTER-POINTS TO MAKE:
{cp_block}

SCIENTIFIC EVIDENCE PER COUNTER-POINT:
{evidence_block}

Task: Write a single, cohesive rebuttal argument that:
  1. Opens by identifying the core weakness in the original argument
  2. Addresses each counter-point with supporting evidence
  3. Exposes the logical flaws naturally within the argument flow
  4. Closes with a strong concluding statement

Requirements:
  - STRUCTURE  : Opening attack -> evidence-backed counter-points -> conclusion
  - EVIDENCE   : Cite or paraphrase the PubMed abstracts where relevant
  - CONFIDENCE : No hedging. Be assertive and direct.
  - TONE       : Formal debate style, not personal
  - LENGTH     : 180-250 words

Write the rebuttal now:"""

    # CHANGED: was _llama_chat(...), now _local_chat(...)
    rebuttal_text = _local_chat(messages=[{"role": "user", "content": prompt}], max_new_tokens=900)

    print(f"\n📝 Final Rebuttal:\n{rebuttal_text}\n")

    return {
        "rebuttal": rebuttal_text,
        "messages": [{"role": "rebuttal", "content": rebuttal_text}],
    }


# ============================================================
# Graph Assembly
# ============================================================


def create_rebuttal_graph():
    workflow = StateGraph(RebuttalState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("pubmed", rebuttal_pubmed_node)
    workflow.add_node("rebuttal", rebuttal_node)

    workflow.add_edge("analyze", "pubmed")
    workflow.add_edge("pubmed", "rebuttal")
    workflow.add_edge("rebuttal", END)

    workflow.set_entry_point("analyze")
    return workflow.compile()


# ============================================================
# Public Interface
# ============================================================


def generate_rebuttal(argument: str, topic: str) -> dict:
    """
    Generate a rebuttal for the given argument on the given topic.

    Note: hf_api_key parameter has been removed — the local Mistral model
    does not require an API key. HF_TOKEN is still read from the environment
    by get_judge_lm if needed for model downloading.
    """
    initialize_runtime_models()

    initial_state: RebuttalState = {
        "original_argument": argument,
        "topic": topic,
        "logical_flaws": [],
        "counter_points": [],
        "pubmed_evidence": {},
        "rebuttal": "",
        "messages": [],
    }

    print("\n" + "=" * 80)
    print("⚔️  DEBATE REBUTTAL GENERATOR")
    print("=" * 80)
    print(f"Topic    : {topic}")
    suffix = "..." if len(argument) > 120 else ""
    print(f"Argument : {argument[:120]}{suffix}")

    final_state = create_rebuttal_graph().invoke(initial_state)

    result = {
        "rebuttal": final_state["rebuttal"],
        "logical_flaws": final_state["logical_flaws"],
        "counter_points": final_state["counter_points"],
        "evidence_retrieved": {
            cp: len(abstracts)
            for cp, abstracts in final_state["pubmed_evidence"].items()
        },
        "messages": final_state["messages"],
    }

    print("\n" + "=" * 80)
    print("✅ FINAL REBUTTAL")
    print("=" * 80)
    print(f"\nLogical Flaws    : {len(result['logical_flaws'])}")
    for flaw in result["logical_flaws"]:
        print(f"  - {flaw}")
    print(f"\nCounter-Points   : {len(result['counter_points'])}")
    for idx, counter_point in enumerate(result["counter_points"], 1):
        print(
            f"  {idx}. {counter_point} "
            f"[{result['evidence_retrieved'].get(counter_point, 0)} abstracts]"
        )
    print(f"\n📝 Rebuttal:\n{result['rebuttal']}")
    print("=" * 80)

    return result


# ============================================================
# CLI entry point
# ============================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a rebuttal argument.")
    parser.add_argument("--topic", type=str, required=True, help="Debate topic text")
    parser.add_argument(
        "--argument",
        type=str,
        required=True,
        help="Argument text to rebut",
    )
    parser.add_argument(
        "--force-model-download",
        action="store_true",
        help="Force re-download of cached runtime models",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.force_model_download:
        get_local_model_dir(_NER_MODEL_ID, force_download=True)

    generate_rebuttal(
        argument=args.argument,
        topic=args.topic,
    )