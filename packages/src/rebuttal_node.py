"""
Debate Rebuttal Generator
Architecture:
  analyze_node -> rebuttal_pubmed_node -> rebuttal_node

Refactor goals:
  - Run on both GPU and CPU (PC-safe fallback)
  - Load runtime models lazily (no heavy import-time initialization)
  - Use local model cache via local_model_store.get_local_model_dir
  - Use d4data/biomedical-ner-all for biomedical NER
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
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, END
from transformers import pipeline as hf_pipeline

try:
    from .local_model_store import get_local_model_dir
except ImportError:
    from local_model_store import get_local_model_dir

load_dotenv()  # Load environment variables from .env file if present
# ============================================================
# Runtime model configuration
# ============================================================

_HAS_CUDA = torch.cuda.is_available()
_HF_DEVICE_ID = int(os.getenv("REBUTTAL_GPU_DEVICE", "0")) if _HAS_CUDA else -1
_NER_MODEL_ID = "d4data/biomedical-ner-all"

# FIX 1: Removed duplicate _ner_pipeline_gene — it was identical to _ner_pipeline,
#         wasting memory/VRAM and causing every text to be processed twice.
_ner_pipeline = None


def initialize_runtime_models(force_reload: bool = False) -> None:
    """Lazily initialize runtime models with local cache and CPU fallback."""
    global _ner_pipeline

    # FIX 2: Guard now checks only the single pipeline variable; the previous
    #         two-variable 'and' guard could skip re-init after a partial failure.
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

    # FIX 1 (continued): Loop now iterates over a single pipeline instead of two
    #                     identical ones.
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
            # FIX 3: Added timeout=15 to prevent the node hanging indefinitely
            #         when NCBI is slow or unreachable.
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

        # FIX 4: Decode the response bytes explicitly as UTF-8 (with error
        #         replacement) before passing to ET.fromstring().  Passing raw
        #         bytes can cause ParseError on abstracts containing accented
        #         characters or extended Unicode.
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
# LLM helper
# ============================================================


def _llama_chat(messages: list, max_tokens: int = 800) -> str:
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is not set.")

    client = InferenceClient(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=api_key,
        timeout=120,
    )

    print(f"   📡 Calling Llama-3-8B-Instruct (max_tokens={max_tokens}) ...")
    resp = client.chat_completion(messages, max_tokens=max_tokens)
    print("   ✅ Response received.")
    return (resp.choices[0].message.content or "").strip()



def _extract_json_block(raw_text: str) -> dict:
    """Robustly parse JSON object from model output."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            # FIX 5: Preserve the original exception as context so callers can
            #         see which parse step failed ('raise inner from exc').
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as inner:
                raise inner from exc
        raise


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

    raw = _llama_chat(messages=[{"role": "user", "content": prompt}], max_tokens=600)
    parsed = _extract_json_block(raw)

    logical_flaws = parsed.get("logical_flaws", [])
    counter_points = parsed.get("counter_points", [])[:3]

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

    rebuttal_text = _llama_chat(messages=[{"role": "user", "content": prompt}], max_tokens=900)

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


def generate_rebuttal(argument: str, topic: str, hf_api_key: str = None) -> dict:
    if hf_api_key:
        os.environ["HF_TOKEN"] = hf_api_key

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
    # FIX 6: Avoid appending "..." when the argument is already shorter than 120
    #         characters — previously produced misleading output like "short arg..."
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
        "--hf-api-key",
        type=str,
        default=None,
        help="Optional HF token override; otherwise uses HF_TOKEN env var",
    )
    parser.add_argument(
        "--force-model-download",
        action="store_true",
        help="Force re-download of cached runtime models",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    hf_token = os.getenv("HF_TOKEN")

    if args.force_model_download:
        get_local_model_dir(_NER_MODEL_ID, force_download=True)

    generate_rebuttal(
        argument=args.argument,
        topic=args.topic,
        hf_api_key=hf_token
    )