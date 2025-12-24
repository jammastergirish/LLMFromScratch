# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece", "tqdm"]
# ///

import torch
import os
from datetime import datetime
from config import ModelConfig, Architecture
from training_args import TransformerTrainingArgs
from trainer import TransformerTrainer
from dataset import TransformerDataset
from model import TransformerModelWithEinops, TransformerModelWithoutEinops

device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# Load training data
with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Architecture selection
MODEL_ARCHITECTURE = "GPT"  # Options: "GPT", "LLAMA"
USE_EINOPS = True  # Use einops versions
USE_SMALL_MODEL = True  # Set to False for full model size

# Initialize config based on architecture
if MODEL_ARCHITECTURE == "LLAMA":
    if USE_SMALL_MODEL:
        cfg = ModelConfig.llama_small()
        print("Using SMALL LLaMA config (faster for Mac)")
    else:
        cfg = ModelConfig.llama_full()
        print("Using FULL LLaMA config")
else:  # GPT
    if USE_SMALL_MODEL:
        cfg = ModelConfig.gpt_small()
        print("Using SMALL GPT config (faster for Mac)")
    else:
        cfg = ModelConfig.gpt_full()
        print("Using FULL GPT config (GPT-2 size)")

# Create dataset
TOKENIZER_TYPE = "character"
dataset = TransformerDataset(text, cfg, tokenizer_type=TOKENIZER_TYPE)
dataset.print_info()

# Get train/val splits
X_train, Y_train = dataset.get_train_data()
X_val, Y_val = dataset.get_val_data()

# Update cfg (dataset updates d_vocab internally)
cfg = dataset.cfg

# Initialize model
if USE_EINOPS:
    model = TransformerModelWithEinops(cfg)
    model_type_str = "with_einops"
else:
    model = TransformerModelWithoutEinops(cfg)
    model_type_str = "without_einops"

model = model.to(device)
print(f"\nInitialized {MODEL_ARCHITECTURE} model ({model_type_str})")
print(f"Model on device: {next(model.parameters()).device}")
print(
    f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Training setup
args = TransformerTrainingArgs()
# Reduce eval_iters and batch_size for faster training on Mac
if USE_SMALL_MODEL:
    args.eval_iters = 50  # Faster evaluation for small model
    args.batch_size = 16  # Smaller batch for Mac memory

# Create timestamped checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
checkpoint_dir = os.path.join("checkpoints", timestamp)
args.save_dir = checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"\nCheckpoints will be saved to: {checkpoint_dir}")

# Create trainer
trainer = TransformerTrainer(
    model=model,
    args=args,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    device=device,
)

# Start training
trainer.train()

print("\n" + "=" * 50)
print("Training complete!")
print("=" * 50)
final_model_path = os.path.join(args.save_dir, "final_model.pt")
print(f"Model saved to: {final_model_path}")
print("\nTo generate text, run:")
print(
    f"  uv run infer.py --checkpoint {final_model_path} --prompt 'Your prompt here'")
