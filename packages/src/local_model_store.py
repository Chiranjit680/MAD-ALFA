from pathlib import Path

from huggingface_hub import snapshot_download


_BASE_DIR = Path(__file__).resolve().parent
_MODEL_STORE_DIR = _BASE_DIR / ".model_store"


def get_local_model_dir(repo_id: str, force_download: bool = False) -> str:
    """Download a Hugging Face model once into workspace cache and return local dir."""
    safe_name = repo_id.replace("/", "__")
    model_dir = _MODEL_STORE_DIR / safe_name
    marker = model_dir / ".complete"

    if force_download or not marker.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        marker.write_text("ok", encoding="utf-8")

    return str(model_dir)
