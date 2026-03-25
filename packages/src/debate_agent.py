"""
Debate Argument Generator Subgraph
Architecture:
  query_gen_node → pubmed_node → reranker_node → generator_node → critic_node → decision_node
       ↑                                               ↑                              │
       │         (nli_score < 0.6, re-retrieve)        │   (writing failure, rewrite) │
       └────────────────────────────────────────────────┴──────────────────────────────┘

FIXES APPLIED
-------------
1. `from collections import OrderedDict` — was missing, caused NameError in _pubmed_fetch.

2. _pubmed_fetch print bug — was `print(f"… {deduplicated} abstracts …")` (printed the
   whole list); fixed to `print(f"… {len(deduplicated)} abstracts …")`.

3. MistralGenerator — complete rewrite to use InferenceClient correctly:
   • InferenceClient.chat_completion() does NOT accept an OpenAI-style `tools` kwarg
     for most HF-hosted models.
   • Tools are now described in the system prompt as a plain-text JSON protocol.
   • The model replies with {"tool": "...", "args": {...}} when it wants to call a tool,
     or with the final argument text when it is done.
   • _parse_tool_call() detects which case applies; _execute_tool() dispatches to the
     real Python functions.
   • Loop runs up to 5 rounds before falling back to the last assistant message.
"""

from typing import TypedDict, Annotated, Literal
import argparse
from pathlib import Path
import operator
import os
import json
import requests
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import torch
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, END
from sentence_transformers import CrossEncoder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline

try:
    from .model_inference import load_models, predict_argument_quality
except ImportError:
    from model_inference import load_models, predict_argument_quality


_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _BASE_DIR / "argument_quality_model_4features.pth"
_HAS_CUDA = torch.cuda.is_available()
_HF_DEVICE_ID = 0 if _HAS_CUDA else -1

# Runtime-loaded models (initialized by initialize_runtime_models)
_ner_pipeline = None
_ner_pipeline_gene = None
_query_tokenizer = None
_query_model = None


# ============================================================
# State Definition
# ============================================================

class DebateState(TypedDict):
    topic: str
    stance: str                  # "PRO" or "CON"
    pubmed_query: str
    argument: str
    iteration: int
    max_iterations: int
    pubmed_abstracts: list
    reranked_evidence: list
    nli_score: float
    semantic_score: float
    logprob_score: float
    confidence_score: float
    quality_score: float
    quality_level: str
    accepted: bool
    feedback: str
    prev_arg: str
    opposition_arg: str
    messages: Annotated[list, operator.add]
    retrieval_feedback: str
    retrieval_attempts: int
    failure_reason: str


# ============================================================
# Real Tool Implementations
# ============================================================

_ddg_search = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(
        region="en-us",
        time="y",
        max_results=5,
    )
)

def web_search(query: str) -> str:
    try:
        result = _ddg_search.run(query)
        return result if result else "No results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"


def pubmed_search_tool(query: str, retmax: int = 5) -> str:
    pmids = (
        requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query,
                    "retmax": retmax, "retmode": "json"},
        )
        .json()
        .get("esearchresult", {})
        .get("idlist", [])
    )
    if not pmids:
        return "No PubMed results found."

    root = ET.fromstring(
        requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        ).content
    )
    abstracts = []
    for article in root.findall(".//PubmedArticle"):
        text = " ".join(
            s.text.strip() for s in article.findall(".//AbstractText") if s.text
        )
        if text:
            abstracts.append(text)

    return "\n\n---\n\n".join(abstracts) if abstracts else "No abstracts found."


def get_rhetorical_devices() -> str:
    return (
        "• Ethos    – Cite credible experts or institutions to build trust\n"
        "• Pathos   – Use vivid, human-impact examples to evoke emotion\n"
        "• Logos    – Build step-by-step logical reasoning chains\n"
        "• Analogy  – Make abstract data relatable via familiar comparisons\n"
        "• Anaphora – Repeat a key phrase for rhythmic emphasis\n"
        "• Concession + Rebuttal – Acknowledge the strongest opposing point, then dismantle it\n"
        "• Statistics first – Lead with a striking number to anchor attention"
    )


# ============================================================
# Reranker utility
# ============================================================

_reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def rerank_top3(query: str, documents: list[str]) -> list[str]:
    if not documents:
        return []
    scores = _reranker_model.predict([(query, doc) for doc in documents])
    return [documents[i] for i in np.argsort(scores)[::-1][:3]]


# ============================================================
# Internal PubMed fetch
# ============================================================

def _ner_extract(text: str) -> list[str]:
    if _ner_pipeline is None or _ner_pipeline_gene is None:
        initialize_runtime_models()

    seen     = set()
    entities = []
    for pipe in [_ner_pipeline, _ner_pipeline_gene]:
        try:
            for ent in pipe(text):
                mention = ent["word"].strip()
                if mention.lower() not in seen and len(mention) > 2 and not mention.startswith("##"):
                    seen.add(mention.lower())
                    entities.append(mention)
        except Exception as e:
            print(f"   ⚠️  NER pipeline error: {e}")
    return entities


def _pubmed_fetch(query: str, retmax: int = 15) -> list[str]:

    def _fetch_by_term(term: str, retmax: int) -> list[str]:
        try:
            pmids = (
                requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db": "pubmed", "term": term,
                            "retmax": retmax, "retmode": "json"},
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
                ).content
            )
            abstracts = []
            for article in root.findall(".//PubmedArticle"):
                text = " ".join(
                    s.text.strip() for s in article.findall(".//AbstractText") if s.text
                )
                if text:
                    abstracts.append(text)
            return abstracts
        except Exception as e:
            print(f"   ⚠️  PubMed fetch failed for term '{term}': {e}")
            return []

    print(f"\n   🔍 [Full query] {query}")
    all_abstracts = _fetch_by_term(query, retmax=retmax)
    print(f"      → {len(all_abstracts)} abstract(s)")

    entities = _ner_extract(query)
    print(f"\n   🧬 NER extracted {len(entities)} entity/entities: {entities}")

    per_entity_retmax = max(3, retmax // max(len(entities), 1))

    for entity in entities:
        print(f"\n   🔍 [NER entity] {entity}")
        entity_abstracts = _fetch_by_term(entity, retmax=per_entity_retmax)
        print(f"      → {len(entity_abstracts)} abstract(s)")
        all_abstracts.extend(entity_abstracts)

    seen = OrderedDict()
    for abstract in all_abstracts:
        key = abstract[:120].strip().lower()
        if key not in seen:
            seen[key] = abstract

    deduplicated = list(seen.values())
    # FIX #2 – was printing the list object instead of its length
    print(f"\n   📚 Total after dedup: {len(deduplicated)} abstracts "
          f"({len(all_abstracts)} raw → {len(deduplicated)} unique)")

    return deduplicated


# ============================================================
# Runtime model initialization
# ============================================================

_QUERY_GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def initialize_runtime_models(force_reload: bool = False) -> None:
    global _ner_pipeline, _ner_pipeline_gene, _query_tokenizer, _query_model

    if not force_reload and all(
        x is not None for x in (_ner_pipeline, _ner_pipeline_gene, _query_tokenizer, _query_model)
    ):
        return

    print("🚀 Initializing runtime models...")
    print(f"   CUDA available: {_HAS_CUDA}")

    _ner_pipeline = hf_pipeline(
        "ner",
        model="pruas/BENT-PubMedBERT-NER-Disease",
        aggregation_strategy="simple",
        device=_HF_DEVICE_ID,
    )
    _ner_pipeline_gene = hf_pipeline(
        "ner",
        model="pruas/BENT-PubMedBERT-NER-Gene",
        aggregation_strategy="simple",
        device=_HF_DEVICE_ID,
    )

    _query_tokenizer = AutoTokenizer.from_pretrained(_QUERY_GEN_MODEL_ID)
    query_dtype = torch.float16 if _HAS_CUDA else torch.float32
    device_map = "cuda" if _HAS_CUDA else "cpu"
    _query_model = AutoModelForCausalLM.from_pretrained(
        _QUERY_GEN_MODEL_ID,
        torch_dtype=query_dtype,
        device_map=device_map,
    )
    _query_model.eval()
    print(f"✅ Query-gen model loaded on {next(_query_model.parameters()).device}")


# ============================================================
# Node 1 – Query Generator
# ============================================================

def query_gen_node(state: DebateState) -> dict:
    if _query_model is None or _query_tokenizer is None:
        initialize_runtime_models()

    print(f"\n{'='*60}")
    print("🧠 QUERY GEN NODE – Generating PubMed search query (local GPU)")
    print(f"{'='*60}")

    topic              = state["topic"]
    stance             = state["stance"]
    retrieval_feedback = state.get("retrieval_feedback", "")
    retrieval_attempts = state.get("retrieval_attempts", 0)

    stance_instruction = (
        "The query must retrieve studies that SUPPORT the topic."
        if stance == "PRO"
        else
        "The query must retrieve studies that are AGAINST the topic."
    )

    feedback_block = ""
    if retrieval_feedback:
        feedback_block = (
            f"\nPREVIOUS RETRIEVAL FAILED (Attempt {retrieval_attempts})\n"
            f"Critic feedback on why the evidence was insufficient:\n"
            f"{retrieval_feedback}\n\n"
            f"You MUST generate a meaningfully different query that addresses this gap.\n"
            f"Try different MeSH terms, synonyms, or a broader/narrower scope.\n"
        )

    prompt = (
        f"You are a biomedical librarian expert at crafting PubMed search queries.\n\n"
        f"Task: Generate a single optimised PubMed search query.\n\n"
        f"Topic : {topic}\n"
        f"Stance: {stance}\n\n"
        f"Stance instruction:\n{stance_instruction}\n"
        f"{feedback_block}\n"
        f"Rules:\n"
        f"- Use MeSH terms and Boolean operators (AND, OR, NOT) where appropriate\n"
        f"- Be specific enough to retrieve highly relevant abstracts\n"
        f"- Output ONLY the raw query string, nothing else\n\n"
        f"Query:"
    )

    try:
        messages  = [{"role": "user", "content": prompt}]
        formatted = _query_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = prompt

    inputs = _query_tokenizer(formatted, return_tensors="pt").to(_query_model.device)

    with torch.no_grad():
        output_ids = _query_model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3 + (0.1 * retrieval_attempts),
            do_sample=True,
            pad_token_id=_query_tokenizer.eos_token_id,
        )

    new_tokens   = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_query    = _query_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pubmed_query = (
        raw_query
        .split("\n")[0]
        .removeprefix("Query:")
        .strip()
        .strip('"')
        .strip("'")
    )

    print(f"\n   Topic             : {topic}")
    print(f"   Stance            : {stance}")
    print(f"   Retrieval attempt : {retrieval_attempts + 1}")
    print(f"   Query             : {pubmed_query}")
    if retrieval_feedback:
        print(f"   Feedback used     : {retrieval_feedback[:100]}...")

    return {
        "pubmed_query":       pubmed_query,
        "retrieval_attempts": retrieval_attempts + 1,
        "retrieval_feedback": "",
        "messages": [{
            "role":    "query_gen",
            "content": f"[Attempt {retrieval_attempts+1}] Generated PubMed query: {pubmed_query}",
        }],
    }


# ============================================================
# Node 2 – PubMed Retrieval
# ============================================================

def pubmed_node(state: DebateState) -> dict:
    print(f"\n{'='*60}")
    print("🔬 PUBMED NODE – Fetching evidence")
    print(f"{'='*60}")

    query = state["pubmed_query"]
    print(f"   Query: {query}")

    abstracts = _pubmed_fetch(query, retmax=15)
    print(f"   Retrieved {len(abstracts)} abstract(s)")

    return {
        "pubmed_abstracts": abstracts,
        "messages": [{"role": "pubmed",
                      "content": f"Retrieved {len(abstracts)} abstracts for query: {query}"}],
    }


# ============================================================
# Node 3 – Reranker
# ============================================================

def reranker_node(state: DebateState) -> dict:
    print(f"\n{'='*60}")
    print("📊 RERANKER NODE – Selecting top-3 evidence")
    print(f"{'='*60}")

    query     = f"{state['topic']} {state['stance']}"
    abstracts = state.get("pubmed_abstracts", [])
    top3      = rerank_top3(query, abstracts)

    print(f"   Input docs : {len(abstracts)}")
    print(f"   Output docs: {len(top3)}")
    for i, doc in enumerate(top3, 1):
        print(f"\n   [Rank {i}] {doc[:120]}...")

    return {
        "reranked_evidence": top3,
        "messages": [{"role": "reranker",
                      "content": f"Top-{len(top3)} evidence selected"}],
    }


# ============================================================
# Node 4 – Generator  ← FIX #3: InferenceClient used correctly
# ============================================================

# Shared tool registry
_TOOL_FUNCTIONS = {
    "web_search":             web_search,
    "pubmed_search_tool":     pubmed_search_tool,
    "get_rhetorical_devices": get_rhetorical_devices,
}

# Tool descriptions embedded in the system prompt.
# HF-hosted Mistral via InferenceClient does NOT support the OpenAI
# `tools` parameter — we implement tool-calling via a JSON protocol instead.
_TOOL_SCHEMA_BLOCK = """
You have access to the following tools.
To call a tool, reply with ONLY a raw JSON object (no markdown, no extra text):
  {"tool": "<name>", "args": {<key: value, ...>}}

Available tools:
  • web_search(query: str)
      Search the web via DuckDuckGo for current stats, news, or policy context.
  • pubmed_search_tool(query: str, retmax: int = 5)
      Fetch peer-reviewed biomedical abstracts from PubMed (retmax ≤ 20).
  • get_rhetorical_devices()
      Returns persuasion techniques (ethos, pathos, logos, anaphora, etc.).

When you have gathered enough evidence and are ready to write the final argument,
reply with the argument text directly — plain prose, NO JSON wrapper.
""".strip()


def _parse_tool_call(text: str) -> dict | None:
    """
    Return a parsed tool-call dict if the model replied with a JSON tool call,
    otherwise return None (the reply is the final argument).
    """
    text = text.strip()
    # Strip optional markdown code fences the model might add
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _execute_tool(call: dict) -> str:
    fn_name = call.get("tool", "")
    args    = call.get("args", {})
    print(f"   🔧 Tool called : {fn_name}({args})")
    if fn_name not in _TOOL_FUNCTIONS:
        return f"Error: unknown tool '{fn_name}'"
    try:
        result = _TOOL_FUNCTIONS[fn_name](**args)
    except Exception as e:
        result = f"Tool error: {e}"
    print(f"   ✅ Result preview: {str(result)[:120]}...")
    return str(result)


class LlamaGenerator:
    """
    Generates debate arguments using meta-llama/Meta-Llama-3-8B-Instruct
    via HuggingFace InferenceClient.chat_completion().

    Tool calling uses a prompt-based JSON protocol:
      1. System prompt describes tools and the {"tool":...,"args":...} call format.
      2. Each round: if the model replies with a JSON tool call, the tool is
         executed and its result is fed back as a user message.
      3. When the model replies with plain prose, that is the final argument.
      4. After 5 rounds without a final answer, the last reply is returned.
    """

    _MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __init__(self):
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("Set HF_TOKEN environment variable")
        self.client = InferenceClient(self._MODEL, token=api_key)

    def generate(
        self,
        topic: str,
        stance: str,
        pubmed_query: str,
        evidence: list,
        feedback: str = "",
        iteration: int = 1,
    ) -> str:

        evidence_block = (
            "\n\n".join(f"[PubMed Evidence {i+1}]: {e}" for i, e in enumerate(evidence))
            or "No PubMed abstracts pre-retrieved for this topic."
        )

        stance_goal = (
            "Your goal is to build a SUPPORTING argument — highlight benefits, "
            "positive outcomes, and evidence in favour of the topic."
            if stance == "PRO"
            else
            "Your goal is to build an OPPOSING argument — highlight risks, harms, "
            "negative outcomes, and evidence against the topic."
        )

        system_prompt = (
            f"You are an expert debate argument generator.\n\n"
            f"Topic : {topic}\n"
            f"Stance: {stance}\n"
            f"{stance_goal}\n\n"
            f"PRE-RETRIEVED SCIENTIFIC EVIDENCE\n"
            f"(fetched using stance-aware PubMed query: \"{pubmed_query}\")\n"
            f"{evidence_block}\n\n"
            f"{_TOOL_SCHEMA_BLOCK}\n\n"
            f"Recommended strategy:\n"
            f"  1. Review the pre-retrieved PubMed abstracts above.\n"
            f"  2. Call web_search for recent statistics or policy context.\n"
            f"  3. Call pubmed_search_tool if deeper scientific backing is needed.\n"
            f"  4. Optionally call get_rhetorical_devices for structure guidance.\n"
            f"  5. Write ONE cohesive, evidence-grounded argument aligned with your stance.\n\n"
            f"Argument requirements:\n"
            f"  RELEVANCE   - Directly address the topic.\n"
            f"  STANCE      - Every sentence must serve your {stance} position.\n"
            f"  EVIDENCE    - Cite or paraphrase real sources.\n"
            f"  LOGIC       - Clear step-by-step reasoning.\n"
            f"  CONFIDENCE  - No hedging. Be assertive.\n"
            f"  STRUCTURE   - Intro -> body -> conclusion.\n"
            f"  LENGTH      - 130-200 words."
        )

        if feedback:
            system_prompt += (
                f"\n\nPREVIOUS CRITIQUE (Iteration {iteration}):\n{feedback}\n"
                "Address every point in this revision."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Generate a compelling {stance} argument for: {topic}"},
        ]

        last_text = ""

        for round_num in range(5):
            print(f"   🔄 Generator round {round_num + 1}")

            resp = self.client.chat_completion(messages, max_tokens=900)

            reply_text = (resp.choices[0].message.content or "").strip()
            last_text  = reply_text

            tool_call = _parse_tool_call(reply_text)

            if tool_call:
                tool_result = _execute_tool(tool_call)
                messages.append({"role": "assistant", "content": reply_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for '{tool_call.get('tool', 'unknown')}':\n"
                        f"{tool_result}\n\n"
                        "Continue: call another tool if needed, or write the final argument."
                    ),
                })
            else:
                print(f"   ✅ Final argument produced on round {round_num + 1}")
                return reply_text

        print("   ⚠️  Max tool rounds reached — returning last assistant reply")
        return last_text


def generator_node(state: DebateState) -> dict:
    print(f"\n{'='*60}")
    print(f"🤖 GENERATOR NODE (Iteration {state['iteration']})")
    print(f"{'='*60}")

    argument = LlamaGenerator().generate(
        topic=state["topic"],
        stance=state["stance"],
        pubmed_query=state["pubmed_query"],
        evidence=state.get("reranked_evidence", []),
        feedback=state.get("feedback", ""),
        iteration=state["iteration"],
    )

    print(f"\n📝 Generated Argument:\n{argument}\n")
    return {
        "argument": argument,
        "messages": [{"role": "generator", "content": argument}],
    }


# ============================================================
# Node 5 – Critic
# ============================================================

def critic_node(state: DebateState) -> dict:
    print(f"\n{'='*60}")
    print("🔍 CRITIC NODE – Running FFN Evaluation")
    print(f"{'='*60}")

    result = predict_argument_quality(
        argument=state["argument"],
        topic=state["topic"],
        stance=state["stance"],
        return_details=True,
    )

    features         = result["features"]
    nli_score        = features["relevance_support"]
    semantic_score   = features["similarity"]
    confidence_score = features["confidence"]

    print(f"\n📊 FFN Quality Score  : {result['quality_score']:.3f}  ({result['quality_level']})")
    print(f"   NLI Relevance      : {nli_score:.3f}")
    print(f"   Semantic Similarity: {semantic_score:.3f}")
    print(f"   Confidence         : {confidence_score:.3f}")
    for name, interp in result["interpretations"].items():
        print(f"   • {name}: {interp}")

    return {
        "quality_score":    result["quality_score"],
        "quality_level":    result["quality_level"],
        "nli_score":        nli_score,
        "semantic_score":   semantic_score,
        "confidence_score": confidence_score,
        "logprob_score":    confidence_score,
        "messages": [{"role": "critic",
                      "content": f"Quality: {result['quality_score']:.3f} ({result['quality_level']})"}],
    }


# ============================================================
# Node 6 – Decision
# ============================================================

def decision_node(state: DebateState) -> dict:
    print(f"\n{'='*60}")
    print("⚖️  DECISION NODE")
    print(f"{'='*60}")

    quality_score      = state["quality_score"]
    nli_score          = state["nli_score"]
    semantic_score     = state["semantic_score"]
    threshold          = 0.80
    retrieval_attempts = state.get("retrieval_attempts", 0)
    max_retrieval      = 2

    print(f"   Quality Score      : {quality_score:.3f}   Threshold : {threshold}")
    print(f"   NLI Score          : {nli_score:.3f}   Threshold : 0.60")
    print(f"   Retrieval attempts : {retrieval_attempts}/{max_retrieval}")

    if quality_score >= threshold:
        print("✅ ACCEPTED")
        return {
            "accepted":       True,
            "feedback":       "",
            "failure_reason": "",
            "messages": [{"role": "decision", "content": "Argument accepted"}],
        }

    if state["iteration"] >= state["max_iterations"]:
        print("⚠️  Max iterations reached – accepting best attempt")
        return {
            "accepted":       True,
            "feedback":       "Max iterations reached",
            "failure_reason": "",
            "messages": [{"role": "decision", "content": "Accepted (max iterations)"}],
        }

    if nli_score < 0.6 and retrieval_attempts < max_retrieval:
        print("🔄 EVIDENCE FAILURE → routing back to query_gen for fresh retrieval")

        retrieval_feedback_parts = [
            f"The argument scored NLI={nli_score:.3f} which is below 0.60.",
            "The retrieved evidence was not relevant enough to the topic and stance.",
        ]
        if semantic_score < 0.6:
            retrieval_feedback_parts.append(
                "Semantic alignment was also weak — try different terminology or MeSH terms."
            )
        retrieval_feedback_parts.append(
            f"Previous failing query was: \"{state['pubmed_query']}\". "
            "Generate a meaningfully different query."
        )

        retrieval_feedback = " ".join(retrieval_feedback_parts)
        print(f"\n📝 Retrieval Feedback:\n{retrieval_feedback}\n")

        return {
            "accepted":           False,
            "failure_reason":     "evidence",
            "retrieval_feedback": retrieval_feedback,
            "feedback":           "",
            "iteration":          state["iteration"] + 1,
            "messages": [{
                "role":    "decision",
                "content": f"Evidence failure (NLI={nli_score:.3f}) → re-retrieving",
            }],
        }

    print("✍️  WRITING FAILURE → routing back to generator for rewrite")

    parts = []
    if nli_score < 0.6:
        parts.append(
            "❌ RELEVANCE: Argument is not clearly related to the topic. "
            "Make explicit connections to the core question even with current evidence."
        )
    if semantic_score < 0.6:
        parts.append(
            "❌ ALIGNMENT: Use key terms and concepts from the topic statement directly."
        )
    if quality_score < 0.5:
        parts.append(
            "❌ OVERALL: Include specific evidence, logical chains, and concrete examples."
        )
    elif quality_score < threshold:
        parts.append(
            f"⚠️  ALMOST THERE: Score {quality_score:.3f} vs threshold {threshold}. "
            "Strengthen your weakest areas."
        )

    feedback = "\n\n".join(parts)
    print(f"\n📝 Writing Feedback:\n{feedback}\n")

    return {
        "accepted":           False,
        "failure_reason":     "writing",
        "feedback":           feedback,
        "retrieval_feedback": "",
        "iteration":          state["iteration"] + 1,
        "messages": [{
            "role":    "decision",
            "content": f"Writing failure → rewriting (iter {state['iteration']+1})",
        }],
    }


# ============================================================
# Routing
# ============================================================

def route_decision(state: DebateState) -> Literal["query_gen", "generator", "end"]:
    if state["accepted"]:
        return "end"
    if state["failure_reason"] == "evidence":
        return "query_gen"
    return "generator"


# ============================================================
# Graph Assembly
# ============================================================

def create_debate_argument_graph():
    workflow = StateGraph(DebateState)

    workflow.add_node("query_gen",  query_gen_node)
    workflow.add_node("pubmed",     pubmed_node)
    workflow.add_node("reranker",   reranker_node)
    workflow.add_node("generator",  generator_node)
    workflow.add_node("critic",     critic_node)
    workflow.add_node("decision",   decision_node)

    workflow.add_edge("query_gen",  "pubmed")
    workflow.add_edge("pubmed",     "reranker")
    workflow.add_edge("reranker",   "generator")
    workflow.add_edge("generator",  "critic")
    workflow.add_edge("critic",     "decision")

    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "query_gen": "query_gen",
            "generator": "generator",
            "end":       END,
        },
    )

    workflow.set_entry_point("query_gen")
    return workflow.compile()


# ============================================================
# Public Interface
# ============================================================

def generate_debate_argument(
    topic: str,
    stance: str = "PRO",
    max_iterations: int = 5,
    hf_api_key: str = None,
) -> dict:
    if hf_api_key:
        os.environ["HF_TOKEN"] = hf_api_key

    initialize_runtime_models()

    initial_state: DebateState = {
        "topic":               topic,
        "stance":              stance.upper(),
        "pubmed_query":        "",
        "argument":            "",
        "iteration":           1,
        "max_iterations":      max_iterations,
        "pubmed_abstracts":    [],
        "reranked_evidence":   [],
        "nli_score":           0.0,
        "semantic_score":      0.0,
        "logprob_score":       0.0,
        "confidence_score":    0.0,
        "quality_score":       0.0,
        "quality_level":       "",
        "accepted":            False,
        "feedback":            "",
        "prev_arg":            "",
        "opposition_arg":      "",
        "messages":            [],
        "retrieval_feedback":  "",
        "retrieval_attempts":  0,
        "failure_reason":      "",
    }

    print("\n" + "="*80)
    print("🎯 DEBATE ARGUMENT GENERATOR")
    print("="*80)
    print(f"Topic            : {topic}")
    print(f"Stance           : {stance}")
    print(f"Max Iterations   : {max_iterations}")
    print(f"Quality Threshold: 0.80")
    print(f"NLI Threshold    : 0.60  (triggers re-retrieval, max 2x)")
    print(f"LLM Tools        : web_search · pubmed_search_tool · get_rhetorical_devices")

    final_state = create_debate_argument_graph().invoke(initial_state)

    result = {
        "argument":           final_state["argument"],
        "pubmed_query":       final_state["pubmed_query"],
        "quality_score":      final_state["quality_score"],
        "quality_level":      final_state["quality_level"],
        "iterations_used":    final_state["iteration"],
        "retrieval_attempts": final_state["retrieval_attempts"],
        "accepted":           final_state["accepted"],
        "scores": {
            "nli":        final_state["nli_score"],
            "semantic":   final_state["semantic_score"],
            "confidence": final_state["confidence_score"],
            "overall":    final_state["quality_score"],
        },
        "evidence_used": final_state["reranked_evidence"],
        "messages":      final_state["messages"],
    }

    print("\n" + "="*80)
    print("✅ FINAL RESULT")
    print("="*80)
    print(f"PubMed Query       : {result['pubmed_query']}")
    print(f"Quality Score      : {result['quality_score']:.3f}  ({result['quality_level']})")
    print(f"Iterations         : {result['iterations_used']}/{max_iterations}")
    print(f"Retrieval attempts : {result['retrieval_attempts']}")
    print(f"\n📝 Final Argument:\n{result['argument']}")
    print("="*80)
    return result


# ============================================================
# Entry Point
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a debate argument with local model scoring.")
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Debate topic text",
    )
    parser.add_argument(
        "--stance",
        type=str,
        default="PRO",
        choices=["PRO", "CON", "pro", "con"],
        help="Argument stance",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum generator/critic iterations",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Local path to argument quality model checkpoint",
    )
    parser.add_argument(
        "--hf-api-key",
        type=str,
        default=None,
        help="Optional HF token override; otherwise uses HF_TOKEN environment variable",
    )
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    print("Loading trained argument quality model...")
    load_models(args.model_path)

    result = generate_debate_argument(
        topic=args.topic,
        stance=args.stance.upper(),
        max_iterations=args.max_iterations,
        hf_api_key=args.hf_api_key,
    )

    print(f"\nDone in {result['iterations_used']} iteration(s) "
          f"| Final quality: {result['quality_score']:.3f} ({result['quality_level']})")