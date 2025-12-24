# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece", "tqdm"]
# ///

"""Inference script for generating text from trained GPT models."""

import argparse
import torch
from config import ModelConfig
from model import TransformerModelWithEinops, TransformerModelWithoutEinops
from tokenizer import CharacterTokenizer, BPETokenizer
from sampler import TransformerSampler


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and config from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, config, checkpoint_dict)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    cfg = checkpoint.get("cfg")
    if cfg is None:
        # Fallback: use default config
        print("Warning: No config in checkpoint, using default")
        cfg = ModelConfig.gpt_small()
    elif isinstance(cfg, dict):
        # Reconstruct config object from dict
        cfg = ModelConfig(**cfg)
    # If it's already a ModelConfig object, use it directly

    # Determine model type from checkpoint or default
    model_type = checkpoint.get("model_type", "with_einops")

    # Initialize model
    if model_type == "with_einops":
        model = TransformerModelWithEinops(cfg)
    else:
        model = TransformerModelWithoutEinops(cfg)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(
        f"Model loaded: {model_type}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    return model, cfg, checkpoint


def main():
    """Main function for inference script."""
    parser = argparse.ArgumentParser(
        description="Generate text from trained GPT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="First Citizen:",
        help="Starting prompt for text generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling (None to disable)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (None to disable)"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="character",
        choices=["character", "bpe"],
        help="Tokenizer type (must match training)"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="training.txt",
        help="Text file for tokenizer initialization (needed for character tokenizer)"
    )

    args = parser.parse_args()

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    model, _, _ = load_model_from_checkpoint(args.checkpoint, device)

    # Create tokenizer (must match training tokenizer)
    if args.tokenizer_type == "character":
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharacterTokenizer(text)
    elif args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {args.tokenizer_type}")

    print(
        f"Using {args.tokenizer_type} tokenizer (vocab size: {tokenizer.vocab_size})")

    # Create sampler
    sampler = TransformerSampler(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Generate text
    print("\n" + "=" * 50)
    print("Generating text...")
    print("=" * 50)
    print(f"Prompt: {args.prompt}")
    print(
        f"Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("-" * 50)

    generated = sampler.sample(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print(f"\nGenerated text:\n{generated}")
    print("=" * 50)


if __name__ == "__main__":
    main()
