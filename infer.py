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
from training_args import TransformerTrainingArgs


def _print_state_dict_warnings(unexpected_keys, missing_keys):
    """Print warnings about state dict mismatches."""
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected key(s) in "
              f"checkpoint (ignored):")
        for key in unexpected_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")

    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing key(s) in checkpoint "
              f"(using random initialization):")
        for key in missing_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and config from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, config, checkpoint_dict)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Allowlist TransformerTrainingArgs for safe loading (PyTorch 2.6+)
    torch.serialization.add_safe_globals([TransformerTrainingArgs])
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

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

    # Load weights with handling for architecture differences
    # (e.g., checkpoint has pos_embed but current model uses ROPE/ALIBI)
    state_dict = checkpoint["model_state_dict"]
    model_state_dict = dict(model.state_dict())

    # Filter state_dict to only include keys that exist in current model
    filtered_state_dict = {}
    missing_keys = []
    unexpected_keys = []

    for key, value in state_dict.items():
        if key in model_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                shape_msg = (f"{key} (shape mismatch: {value.shape} vs "
                             f"{model_state_dict[key].shape})")
                unexpected_keys.append(shape_msg)
        else:
            unexpected_keys.append(key)

    # Find missing keys
    for key in model_state_dict:
        if key not in filtered_state_dict:
            missing_keys.append(key)

    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)

    # Warn about mismatches
    _print_state_dict_warnings(unexpected_keys, missing_keys)

    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {model_type}, {param_count:.2f}M parameters")
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
        default=None,
        choices=["character", "bpe"],
        help="Tokenizer type (auto-detected from checkpoint if not provided)"
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
    model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    # Get tokenizer type from checkpoint or args
    tokenizer_type = checkpoint.get("tokenizer_type")
    if tokenizer_type is None:
        # Fallback to command-line argument
        if args.tokenizer_type is None:
            raise ValueError(
                "Tokenizer type not found in checkpoint and not provided. "
                "Please specify --tokenizer_type (character or bpe)"
            )
        tokenizer_type = args.tokenizer_type
        print(
            f"Warning: Tokenizer type not in checkpoint, using provided: {tokenizer_type}")
    else:
        # Check if user provided a different tokenizer type
        if args.tokenizer_type is not None and args.tokenizer_type != tokenizer_type:
            print(
                f"Warning: Checkpoint uses tokenizer type '{tokenizer_type}', "
                f"but you provided '{args.tokenizer_type}'. "
                f"Using '{tokenizer_type}' from checkpoint."
            )

    # Create tokenizer (must match training tokenizer)
    if tokenizer_type == "character":
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharacterTokenizer(text)
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    print(
        f"Using {tokenizer_type} tokenizer (vocab size: {tokenizer.vocab_size})")

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
