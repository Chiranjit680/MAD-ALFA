# MAD-ALFA

Local debate argument generator with PubMed retrieval, LLM generation, and argument-quality scoring.

## Local Setup (Windows + NVIDIA GPU)

1. Create and activate a virtual environment.
2. Install PyTorch CUDA build first (recommended CUDA 12.1 wheels):

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3. Install project dependencies:

```powershell
pip install -r requirements.txt
```

4. Set Hugging Face token (required for generator model inference):

```powershell
$env:HF_TOKEN="<your_hf_token>"
```

## GPU Verification

Run this before executing the full pipeline:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run the Debate Agent

From packages/src:

```powershell
python debate_agent.py --topic "Should schools ban smartphones in classrooms?" --stance PRO --max-iterations 5
```

Optional arguments:
- `--model-path` to override local checkpoint path (default is `packages/src/argument_quality_model_4features.pth`)
- `--hf-api-key` to pass token directly (environment variable `HF_TOKEN` is preferred)

## Run Model Inference Only

From packages/src:

```powershell
python model_inference.py --argument "Climate action is urgent" --topic "We should take action on climate change" --stance PRO --details
```

## Run Rebuttal Node

From packages/src:

```powershell
python rebuttal_node.py --topic "Should corticosteroids be used routinely in septic shock management?" --argument "Corticosteroids should be administered routinely in all septic shock patients because they rapidly resolve haemodynamic instability and reduce vasopressor dependency."
```

How cached model loading works:
- Rebuttal NER uses `d4data/biomedical-ner-all`.
- The model is cached in `packages/src/.model_store` via `local_model_store.py`.
- After first download, next runs reuse the local cache automatically.

Optional arguments:
- `--hf-api-key` to pass token directly (otherwise `HF_TOKEN` is used)
- `--force-model-download` to refresh the cached NER model

GPU selection:
- CUDA is used automatically when available.
- To pick a specific GPU, set `REBUTTAL_GPU_DEVICE` before running.

```powershell
$env:REBUTTAL_GPU_DEVICE="0"
python rebuttal_node.py --topic "..." --argument "..."
```