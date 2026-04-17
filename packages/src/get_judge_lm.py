import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ============================================================
# Environment setup
# ============================================================
os.environ["CUDA_LAUNCH_BLOCKING"]              = "0"
os.environ["CUDA_DEVICE_ORDER"]                 = "PCI_BUS_ID"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
_HAS_CUDA = torch.cuda.is_available()

# ============================================================
# Model registry — all models live under one root folder
# Update MODEL_STORE_ROOT or set MODEL_STORE_ROOT in your .env
# ============================================================
MODEL_STORE_ROOT = os.getenv("MODEL_STORE_ROOT", ".model_store")

MODEL_REGISTRY: dict[str, str] = {
    "mistral_7b":  "mistral_7b",
    "qwen_light":  "Qwen__Qwen2.5-0.5B-Instruct",
    "qwen3_8b":    "qwen3_8b",
}

# ============================================================
# Resolve which model to use
# ============================================================
model_key = "qwen_light"

if model_key not in MODEL_REGISTRY:
    raise ValueError(
        f"Unknown model key '{model_key}'. "
        f"Available: {list(MODEL_REGISTRY.keys())}"
    )

SAVE_PATH = os.path.join(MODEL_STORE_ROOT, MODEL_REGISTRY[model_key])

print(f"Selected model : {model_key}")
print(f"Local path     : {SAVE_PATH}")

# ============================================================
# Module-level singletons (lazy-loaded)
# ============================================================
_tokenizer: AutoTokenizer        | None = None
_model:     AutoModelForCausalLM | None = None

# ============================================================
# Helpers
# ============================================================
def _assert_model_exists(path: str) -> None:
    """
    Hard-fail early if the model folder or required files are missing.
    No downloads — local only.
    """
    required = ["config.json", "tokenizer_config.json"]
    missing_files = [f for f in required if not os.path.exists(os.path.join(path, f))]

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Model folder not found: '{path}'\n"
            f"Check MODEL_STORE_ROOT ('{MODEL_STORE_ROOT}') and MODEL ('{model_key}') "
            f"in your .env, or update MODEL_REGISTRY in this file."
        )
    if missing_files:
        raise FileNotFoundError(
            f"Model folder '{path}' exists but looks incomplete. "
            f"Missing: {missing_files}"
        )

# ============================================================
# Load
# ============================================================
def get_lm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Loads tokenizer + model from local disk.
    Raises FileNotFoundError immediately if the path is missing or incomplete.
    """
    _assert_model_exists(SAVE_PATH)

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        SAVE_PATH,
        local_files_only=True,      # never reach out to HF
        # Respect externally pinned CUDA visibility (e.g. CUDA_VISIBLE_DEVICES in
        # debate_benchmarking_hf.py). After masking, selected GPU is local cuda:0.
        device_map={"": 0} if _HAS_CUDA else "cpu",
        torch_dtype=torch.float16 if _HAS_CUDA else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    # Guard: tokenizer vocab must not exceed model embedding table
    embed_size  = int(model.get_input_embeddings().num_embeddings)
    token_count = len(tokenizer)
    if token_count > embed_size:
        raise ValueError(
            f"Tokenizer/model mismatch: tokenizer has {token_count} tokens but "
            f"model embedding table only has {embed_size} entries. "
            f"Check that all files in '{SAVE_PATH}' belong to the same model."
        )

    model.eval()
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    return tokenizer, model


# ============================================================
# Inference
# ============================================================
def model_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Run inference on an already-loaded model."""
    do_sample = temperature > 0.0   # keep temperature & do_sample coherent

    device = next(model.parameters()).device
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    # Early check: catch bad token ids before they hit the CUDA kernel
    vocab_limit  = int(model.get_input_embeddings().num_embeddings)
    max_token_id = int(inputs["input_ids"].max().item())
    if max_token_id >= vocab_limit:
        raise ValueError(
            f"Token id {max_token_id} is out of range for embedding size {vocab_limit}. "
            f"Tokenizer and model weights in '{SAVE_PATH}' may not match."
        )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    try:
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

    except RuntimeError as exc:
        if "device-side assert" not in str(exc).lower():
            raise

        print("⚠️  CUDA device-side assert — retrying once on CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use a local CPU copy; move model back to GPU afterwards
        cpu_model  = model.to("cpu")
        cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = cpu_model.generate(
                **cpu_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        model.to(device)    # restore GPU for future calls

    # Decode only newly generated tokens (strip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated  = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def llm_inference(input_text: str, max_new_tokens: int = 600) -> str:
    """Public entry point: lazily loads the model on first call, then infers."""
    global _tokenizer, _model
    print(f"Running inference with model key '{model_key}' from '{SAVE_PATH}' …")
    if _tokenizer is None or _model is None:
        _tokenizer, _model = get_lm()
    return model_inference(_model, _tokenizer, input_text, max_new_tokens)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    tokenizer, model = get_lm()

    test_prompt = "What is chronic rhinosinusitis?"
    print(f"\n🧪 Test prompt : {test_prompt}")
    response = model_inference(model, tokenizer, test_prompt)
    print(f"📝 Response    : {response}")