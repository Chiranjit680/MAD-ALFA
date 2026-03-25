"""
Simple Inference Function for 4-Feature Argument Quality Model
Easy-to-use function without needing the full class
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# ============================================================
# Model Architecture (must match training)
# ============================================================

class ArgumentQualityModel(nn.Module):
    def __init__(self, input_dim: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            
            nn.Linear(8, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# Global Variables (loaded once, reused)
# ============================================================

_MODEL = None
_NORM_STATS = None
_NLI_PIPELINE = None
_SIMILARITY_MODEL = None
_DEVICE = None
_MODEL_PATH = None

_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _BASE_DIR / "argument_quality_model_4features.pth"


def load_models(model_path: str = str(DEFAULT_MODEL_PATH)):
    """
    Load models into memory (call once at startup)
    
    Args:
        model_path: Path to trained model checkpoint
    """
    global _MODEL, _NORM_STATS, _NLI_PIPELINE, _SIMILARITY_MODEL, _DEVICE, _MODEL_PATH
    
    print("🚀 Loading models for inference...")
    
    # Resolve model path relative to this file if needed
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        resolved_path = (_BASE_DIR / resolved_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved_path}")

    _MODEL_PATH = str(resolved_path)

    # Set device
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {_DEVICE}")
    if _DEVICE.type != "cuda":
        print("   Note: CUDA not detected; running on CPU. Install CUDA-enabled PyTorch for GPU acceleration.")
    
    # Load model checkpoint
    print(f"   Loading model from: {_MODEL_PATH}")
    checkpoint = torch.load(_MODEL_PATH, map_location=_DEVICE)
    
    _NORM_STATS = checkpoint['normalization_stats']
    
    # Initialize and load model
    _MODEL = ArgumentQualityModel(input_dim=4)
    _MODEL.load_state_dict(checkpoint['model_state_dict'])
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    
    # Load NLI pipeline
    print("   Loading NLI pipeline...")
    device_id = 0 if _DEVICE.type == 'cuda' else -1
    _NLI_PIPELINE = pipeline(
        "text-classification",
        model="facebook/bart-large-mnli",
        device=device_id
    )
    
    # Load similarity model
    print("   Loading similarity model...")
    _SIMILARITY_MODEL = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=_DEVICE.type
    )
    
    print("✅ Models loaded successfully!\n")


def calculate_confidence_score(text: str) -> float:
    """Calculate confidence score based on linguistic markers"""
    text_lower = text.lower()
    
    hedging_words = [
        'maybe', 'perhaps', 'might', 'could', 'possibly', 
        'probably', 'likely', 'seems', 'appears', 'i think',
        'i believe', 'i guess', 'uncertain', 'may', 'would'
    ]
    
    words = len(text_lower.split())
    if words == 0:
        return 0.5
    
    hedge_count = sum(1 for word in hedging_words if word in text_lower)
    hedge_ratio = hedge_count / words
    confidence = max(0.0, min(1.0, 1.0 - (hedge_ratio * 10)))
    
    return confidence


def extract_features(argument: str, topic: str, stance: str) -> torch.Tensor:
    """
    Extract 4 features from argument
    
    Args:
        argument: The argument text
        topic: The debate topic
        stance: "PRO" or "CON"
    
    Returns:
        Feature tensor [4]
    """
    if _NLI_PIPELINE is None or _SIMILARITY_MODEL is None:
        raise RuntimeError("Models not loaded! Call load_models() first.")
    
    # Convert stance to binary
    stance_int = 1 if stance.upper() == "PRO" else -1
    stance_binary = (stance_int + 1) / 2
    
    # NLI: relevance to topic
    relevance_result = _NLI_PIPELINE(f"{topic} [SEP] {argument}")[0]
    
    if relevance_result["label"] == "ENTAILMENT":
        relevance_score_support = relevance_result["score"]
    elif relevance_result["label"] == "CONTRADICTION":
        relevance_score_support = 0.0
    else:
        relevance_score_support = 1 - relevance_result["score"]
    
    # Semantic similarity
    embeddings = _SIMILARITY_MODEL.encode([topic, argument], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    similarity_score = (cosine_score.item() + 1) / 2
    
    # Confidence
    confidence_score = calculate_confidence_score(argument)
    
    # Create feature tensor
    features = torch.tensor([
        relevance_score_support,
        similarity_score,
        confidence_score,
        stance_binary
    ], dtype=torch.float32)
    
    return features


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Apply z-score normalization using saved stats"""
    if _NORM_STATS is None:
        raise RuntimeError("Normalization stats not loaded! Call load_models() first.")
    
    normalized = features.clone()
    
    for i in range(len(features)):
        stats = _NORM_STATS[f'feature_{i}']
        mean = stats['mean']
        std = stats['std']
        
        if std > 1e-6:
            normalized[i] = (features[i] - mean) / std
        else:
            normalized[i] = features[i] - mean
    
    return normalized


# ============================================================
# Main Inference Function
# ============================================================

def predict_argument_quality(
    argument: str,
    topic: str,
    stance: str = "PRO",
    return_details: bool = False
) -> Dict:
    """
    Predict the quality of an argument
    
    Args:
        argument: The argument text to evaluate
        topic: The debate topic
        stance: "PRO" or "CON" 
        return_details: If True, return feature breakdown
    
    Returns:
        Dictionary with:
        - quality_score: Predicted quality [0, 1]
        - relevance_score: Average of NLI and similarity
        - quality_level: Text description (Excellent/Good/Fair/Poor)
        - features: Individual feature scores (if return_details=True)
    
    Example:
        >>> result = predict_argument_quality(
        ...     "Climate change causes rising sea levels",
        ...     "We should take action on climate change",
        ...     "PRO"
        ... )
        >>> print(f"Quality: {result['quality_score']:.2f}")
        Quality: 0.78
    """
    
    # Check if models are loaded
    if _MODEL is None:
        raise RuntimeError("Models not loaded! Call load_models() first.")
    
    # Validate inputs
    if not argument or not topic:
        raise ValueError("Both argument and topic must be provided")
    
    if stance.upper() not in ["PRO", "CON"]:
        raise ValueError("Stance must be 'PRO' or 'CON'")
    
    # Extract features
    features = extract_features(argument, topic, stance)
    
    # Normalize features
    normalized_features = normalize_features(features)
    
    # Predict
    with torch.no_grad():
        normalized_features = normalized_features.unsqueeze(0).to(_DEVICE)
        output = _MODEL(normalized_features)
        quality_score = output.item()
    
    # Clip to [0, 1] range
    quality_score = max(0.0, min(1.0, quality_score))
    
    # Calculate relevance score
    relevance_score = (features[0].item() + features[1].item()) / 2
    
    # Determine quality level
    if quality_score >= 0.8:
        quality_level = "Excellent"
    elif quality_score >= 0.6:
        quality_level = "Good"
    elif quality_score >= 0.4:
        quality_level = "Fair"
    elif quality_score >= 0.2:
        quality_level = "Poor"
    else:
        quality_level = "Very Poor"
    
    # Build result
    result = {
        'quality_score': quality_score,
        'relevance_score': relevance_score,
        'quality_level': quality_level,
        'stance': stance.upper()
    }
    
    if return_details:
        feature_names = [
            "relevance_support", "similarity", "confidence", "stance_binary"
        ]
        result['features'] = {
            name: features[i].item() 
            for i, name in enumerate(feature_names)
        }
        result['interpretations'] = {
            'relevance': _interpret_relevance(features[0].item()),
            'similarity': _interpret_similarity(features[1].item()),
            'confidence': _interpret_confidence(features[2].item())
        }
    
    return result


def _interpret_relevance(score: float) -> str:
    """Interpret relevance score"""
    if score < 0.3:
        return "Not relevant to topic"
    elif score >= 0.7:
        return "Highly relevant to topic"
    else:
        return "Somewhat relevant"


def _interpret_similarity(score: float) -> str:
    """Interpret similarity score"""
    if score < 0.5:
        return "Semantically distant from topic"
    elif score >= 0.7:
        return "Closely aligned with topic"
    else:
        return "Moderately aligned with topic"


def _interpret_confidence(score: float) -> str:
    """Interpret confidence score"""
    if score < 0.4:
        return "Too much hedging"
    elif score >= 0.7:
        return "Confident and assertive"
    else:
        return "Moderate confidence"


# ============================================================
# Batch Inference
# ============================================================

def predict_batch(
    arguments: list,
    topics: list,
    stances: list
) -> list:
    """
    Predict quality for multiple arguments at once
    
    Args:
        arguments: List of argument texts
        topics: List of topics
        stances: List of stances ("PRO" or "CON")
    
    Returns:
        List of prediction dictionaries
    
    Example:
        >>> results = predict_batch(
        ...     ["Arg 1", "Arg 2"],
        ...     ["Topic 1", "Topic 2"],
        ...     ["PRO", "CON"]
        ... )
        >>> for r in results:
        ...     print(f"Score: {r['quality_score']:.2f}")
    """
    if not (len(arguments) == len(topics) == len(stances)):
        raise ValueError("All input lists must have the same length")
    
    results = []
    for arg, topic, stance in zip(arguments, topics, stances):
        result = predict_argument_quality(arg, topic, stance)
        results.append(result)
    
    return results


# ============================================================
# CLI Helpers
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Argument quality inference (local GPU/CPU).")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to argument quality checkpoint",
    )
    parser.add_argument("--argument", type=str, help="Argument text")
    parser.add_argument("--topic", type=str, help="Debate topic text")
    parser.add_argument(
        "--stance",
        type=str,
        default="PRO",
        choices=["PRO", "CON", "pro", "con"],
        help="Argument stance",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include feature-level breakdown in output",
    )
    parser.add_argument(
        "--batch-demo",
        action="store_true",
        help="Run small built-in batch demo",
    )
    return parser


def _run_default_demo() -> None:
    print("No --argument/--topic provided; running default single-example demo.")
    result = predict_argument_quality(
        argument="Taking action on climate change reduces long-term economic and health risks.",
        topic="We should take action on climate change",
        stance="PRO",
        return_details=True,
    )
    print(f"Quality score : {result['quality_score']:.3f}")
    print(f"Quality level : {result['quality_level']}")
    print(f"Relevance     : {result['relevance_score']:.3f}")
    print("Features:")
    for name, value in result["features"].items():
        print(f"  - {name}: {value:.3f}")


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print("=" * 80)
    print("SIMPLE INFERENCE FUNCTION")
    print("=" * 80)

    load_models(args.model_path)

    if args.batch_demo:
        batch_results = predict_batch(
            arguments=[
                "Climate change causes rising sea levels",
                "Pizza is delicious",
                "We should act now",
            ],
            topics=[
                "We should take action on climate change",
                "We should take action on climate change",
                "We should take action on climate change",
            ],
            stances=["PRO", "PRO", "PRO"],
        )
        for i, result in enumerate(batch_results, 1):
            print(f"Argument {i}: {result['quality_score']:.3f} ({result['quality_level']})")
    else:
        if not args.argument or not args.topic:
            _run_default_demo()
            print("Tip: pass --argument and --topic for custom input, or --batch-demo for multiple samples.")
            raise SystemExit(0)

        result = predict_argument_quality(
            argument=args.argument,
            topic=args.topic,
            stance=args.stance.upper(),
            return_details=args.details,
        )

        print(f"Quality score : {result['quality_score']:.3f}")
        print(f"Quality level : {result['quality_level']}")
        print(f"Relevance     : {result['relevance_score']:.3f}")
        if args.details:
            print("Features:")
            for name, value in result["features"].items():
                print(f"  - {name}: {value:.3f}")