"""Fine-tuning page for supervised fine-tuning (SFT)."""

import os
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from pretraining.tokenization.tokenizer import (
    CharacterTokenizer,
    SimpleBPETokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
)
from finetuning.data.sft_dataset import SFTDataset
from finetuning.training.sft_trainer import SFTTrainer
from finetuning.training.finetuning_args import FinetuningArgs
from finetuning.training.sft_training_ui import train_sft_model_thread
from pretraining.training.training_ui import initialize_training_state
from ui_components import (
    render_checkpoint_selector, render_finetuning_equations, render_finetuning_code_snippets,
    format_elapsed_time, get_total_training_time, render_training_metrics,
    render_all_losses_graph, render_eval_losses_graph, render_completed_training_ui,
    render_active_training_ui, display_training_status
)
from config import PositionalEncoding


def _extend_positional_embeddings(pos_embed_module, new_max_length: int):
    """
    Extend positional embeddings to support longer sequences.

    Uses interpolation to extend the embedding matrix. This is a common
    technique for extending pre-trained positional embeddings to longer contexts.

    Args:
        pos_embed_module: The positional embedding module (PosEmbedWithEinops or PosEmbedWithoutEinops)
        new_max_length: New maximum sequence length
    """
    import torch

    old_W_pos = pos_embed_module.W_pos  # [old_n_ctx, d_model]
    old_n_ctx, d_model = old_W_pos.shape

    if new_max_length <= old_n_ctx:
        # No extension needed
        return

    # Create new embedding matrix
    new_W_pos = torch.empty((new_max_length, d_model),
                            device=old_W_pos.device, dtype=old_W_pos.dtype)

    # Copy existing embeddings
    new_W_pos[:old_n_ctx] = old_W_pos

    # For positions beyond the original length, use interpolation
    # Method: Use the last few positions to extrapolate smoothly
    if old_n_ctx >= 2:
        # Use the trend from the last few positions
        # Compute average "velocity" (difference between consecutive positions)
        # and extrapolate
        last_few = min(10, old_n_ctx)  # Use last 10 positions or all if fewer
        recent_embeds = old_W_pos[-last_few:]  # [last_few, d_model]

        # Compute average change per position
        if last_few >= 2:
            diffs = recent_embeds[1:] - \
                recent_embeds[:-1]  # [last_few-1, d_model]
            # [d_model] - average change per position
            avg_diff = diffs.mean(dim=0)
        else:
            avg_diff = torch.zeros_like(old_W_pos[-1])

        # Extrapolate: start from last position and add scaled differences
        last_embed = old_W_pos[-1]  # [d_model]
        for i in range(old_n_ctx, new_max_length):
            # Scale the difference based on how far we are from the original range
            # Use a decay factor to prevent embeddings from growing too large
            steps_from_end = i - old_n_ctx + 1
            decay = 0.9 ** steps_from_end  # Exponential decay
            new_W_pos[i] = last_embed + avg_diff * steps_from_end * decay
    else:
        # If only one position, just repeat it
        new_W_pos[old_n_ctx:] = old_W_pos[-1]

    # Update the parameter
    pos_embed_module.W_pos = torch.nn.Parameter(new_W_pos)
    # Update cfg.n_ctx to reflect the new max length
    pos_embed_module.cfg.n_ctx = new_max_length


def _start_finetuning_workflow(
    selected_checkpoint_path,
    uploaded_csv,
    batch_size,
    lr,
    weight_decay,
    epochs,
    max_steps_per_epoch,
    eval_interval,
    save_interval,
    max_length,
    use_lora,
    lora_rank,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
):
    """Start the fine-tuning workflow."""
    device = st.session_state.get_device()

    # Load pre-trained model
    with st.spinner("Loading pre-trained model..."):
        model, cfg, checkpoint = st.session_state.load_model_from_checkpoint(
            selected_checkpoint_path, device
        )
        model.train()  # Set to training mode

        # Check sequence length compatibility
        model_max_length = cfg.n_ctx if hasattr(cfg, 'n_ctx') else 256
        if max_length > model_max_length:
            # Check if model uses learned positional embeddings (GPT)
            uses_learned_pos = (
                hasattr(cfg, 'positional_encoding') and
                cfg.positional_encoding == PositionalEncoding.LEARNED
            )

            if uses_learned_pos:
                # Extend positional embeddings if needed
                if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                    _extend_positional_embeddings(model.pos_embed, max_length)
                    st.info(
                        f"✅ Extended positional embeddings from {model_max_length} to {max_length} tokens. "
                        f"Model can now handle sequences up to {max_length} tokens."
                    )
                else:
                    st.error(
                        f"Model was trained with max sequence length {model_max_length}, "
                        f"but fine-tuning requested {max_length}. "
                        f"Please set max_length to {model_max_length} or less."
                    )
                    st.stop()
            else:
                # RoPE and ALiBi can handle longer sequences, but warn if significantly longer
                if max_length > model_max_length * 2:
                    st.warning(
                        f"⚠️ Model was trained with max sequence length {model_max_length}, "
                        f"but fine-tuning will use {max_length}. "
                        f"RoPE/ALiBi can handle longer sequences, but performance may degrade."
                    )
                else:
                    st.info(
                        f"ℹ️ Model context length: {model_max_length}, Fine-tuning length: {max_length}. "
                        f"Using {cfg.positional_encoding.value if hasattr(cfg, 'positional_encoding') else 'positional encoding'} which supports variable length sequences."
                    )

    # Apply LoRA if selected
    if use_lora:
        with st.spinner("Applying LoRA adapters..."):
            from finetuning.peft.lora_utils import convert_model_to_lora, count_lora_parameters
            model = convert_model_to_lora(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            param_counts = count_lora_parameters(model)
            st.success(
                f"✅ LoRA applied! Training {param_counts['lora']:,} LoRA parameters "
                f"({param_counts['lora']/param_counts['total']*100:.2f}% of total). "
                f"{param_counts['frozen']:,} base parameters frozen."
            )

    tokenizer_type = checkpoint.get("tokenizer_type", "character")

    # Create tokenizer (must match pre-trained model)
    if tokenizer_type == "character":
        if os.path.exists("training.txt"):
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            tokenizer = CharacterTokenizer(text)
        else:
            st.error("training.txt not found. Cannot load character tokenizer.")
            st.stop()
    elif tokenizer_type == "bpe-simple":
        if os.path.exists("training.txt"):
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 1000
            tokenizer = SimpleBPETokenizer(text, vocab_size=vocab_size)
        else:
            st.error(
                "training.txt not found. Cannot recreate Simple BPE tokenizer.")
            st.stop()
    elif tokenizer_type == "bpe-tiktoken" or tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    elif tokenizer_type == "sentencepiece":
        if os.path.exists("training.txt"):
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 10000
            tokenizer = SentencePieceTokenizer(text, vocab_size=vocab_size)
        else:
            st.error(
                "training.txt not found. Cannot recreate SentencePiece tokenizer.")
            st.stop()
    else:
        st.error(f"Tokenizer type {tokenizer_type} not supported.")
        st.stop()

    # Save CSV temporarily (use default if no upload)
    csv_path = "temp_sft_data.csv"
    if uploaded_csv:
        with open(csv_path, "wb") as f:
            f.write(uploaded_csv.getbuffer())
        df = pd.read_csv(csv_path)
        st.info(f"Loaded CSV with {len(df)} rows.")
    elif os.path.exists("finetuning.csv"):
        # Use default file
        import shutil
        shutil.copy("finetuning.csv", csv_path)
        df = pd.read_csv(csv_path)
        st.info(f"Using default finetuning.csv with {len(df)} rows.")
    else:
        st.error("No CSV file provided and finetuning.csv not found.")
        st.stop()

    # Create SFT dataset
    with st.spinner("Creating fine-tuning dataset..."):
        dataset = SFTDataset(
            csv_path=csv_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dataset.print_info()

    X_train, Y_train, masks_train = dataset.get_train_data()
    X_val, Y_val, masks_val = dataset.get_val_data()

    # Create save directory: checkpoints/{timestamp}/sft/
    checkpoint_dir = Path(selected_checkpoint_path).parent
    timestamp = checkpoint_dir.name
    sft_dir = checkpoint_dir / "sft"
    sft_dir.mkdir(exist_ok=True)

    # Training args
    training_args = FinetuningArgs(
        batch_size=batch_size,
        epochs=epochs,
        max_steps_per_epoch=max_steps_per_epoch,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=str(sft_dir),
        save_interval=save_interval,
        eval_iters=50,
        use_lora=use_lora,
        lora_rank=lora_rank if use_lora else 8,
        lora_alpha=lora_alpha if use_lora else 8.0,
        lora_dropout=lora_dropout if use_lora else 0.0,
        lora_target_modules=lora_target_modules if use_lora else "all",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        X_train=X_train,
        Y_train=Y_train,
        masks_train=masks_train,
        X_val=X_val,
        Y_val=Y_val,
        masks_val=masks_val,
        device=device,
        eval_interval=eval_interval,
        tokenizer_type=tokenizer_type,
    )

    # Initialize training state
    st.session_state.shared_loss_data = {
        "iterations": [], "train_losses": [], "val_losses": []
    }
    st.session_state.shared_training_logs.clear()
    st.session_state.training_active = True
    st.session_state.trainer = trainer
    st.session_state.training_start_time = time.time()

    training_active_flag = [True]
    progress_data = {
        "iter": 0,
        "loss": 0.0,
        "running_loss": 0.0,
        "val_loss": None,
        "progress": 0.0,
        "all_losses": {
            "iterations": [],
            "current_losses": [],
            "running_losses": []
        }
    }

    # Start training thread
    thread = threading.Thread(
        target=train_sft_model_thread,
        args=(
            trainer,
            st.session_state.shared_loss_data,
            st.session_state.shared_training_logs,
            training_active_flag,
            st.session_state.training_lock,
            progress_data
        ),
        daemon=True
    )
    thread.start()
    st.session_state.training_thread = thread
    st.session_state.training_active_flag = training_active_flag
    st.session_state.progress_data = progress_data

    st.success("Fine-tuning started! Check the visualization below.")
    time.sleep(0.5)
    st.rerun()


def _render_quick_stats(batch_size, lr, epochs, max_length, use_lora, lora_rank=None):
    """Render quick statistics about the fine-tuning configuration."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Batch Size", batch_size)
    with col2:
        st.metric("Learning Rate", f"{lr:.6f}")
    with col3:
        st.metric("Epochs", epochs)
    with col4:
        st.metric("Max Length", max_length)

    if use_lora and lora_rank:
        st.info(
            f"💡 Using LoRA with rank {lora_rank} (parameter-efficient fine-tuning)")


st.title("🎯 Fine-Tuning (SFT)")

# Initialize training state
initialize_training_state()

# Checkpoint selection
with st.container():
    selected_checkpoint = render_checkpoint_selector(
        header="1. Select Pre-Trained Checkpoint",
        filter_finetuned=True,
        help_text="Select a pre-trained model to fine-tune",
        no_checkpoints_message="No pre-trained checkpoints found. Please pre-train a model first."
    )
    st.divider()

# Fine-tuning method selection
with st.container():
    st.markdown("### 🔧 2. Fine-Tuning Method")
    fine_tuning_method = st.radio(
        "Select fine-tuning method",
        ["Full Parameter Fine-Tuning", "LoRA (Parameter-Efficient)"],
        help="Full Parameter: Updates all model weights. LoRA: Only trains small adapter matrices (faster, less memory)."
    )
    st.divider()

# LoRA options (only show if LoRA selected)
use_lora = fine_tuning_method == "LoRA (Parameter-Efficient)"
lora_rank = 8
lora_alpha = 8.0
lora_dropout = 0.0
lora_target_modules = "all"

if use_lora:
    with st.expander("LoRA Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            lora_rank = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                max_value=128,
                value=8,
                help="Dimension of low-rank matrices. Higher = more parameters, more capacity. Typical: 4-16"
            )
            lora_alpha = st.number_input(
                "LoRA Alpha (α)",
                min_value=1.0,
                max_value=256.0,
                value=8.0,
                help="Scaling factor. Typically set equal to rank. Higher = stronger LoRA influence."
            )
        with col2:
            lora_dropout = st.number_input(
                "LoRA Dropout",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                format="%.2f",
                help="Dropout rate for LoRA adapters. Helps prevent overfitting."
            )
            lora_target_modules = st.selectbox(
                "Target Modules",
                ["all", "attention", "mlp"],
                help="Which layers to apply LoRA to. 'all' applies to both attention and MLP layers."
            )

        st.info(
            f"💡 LoRA will train ~{lora_rank * 2} parameters per weight matrix "
            f"(rank={lora_rank}). This is much smaller than full fine-tuning!"
        )

# CSV upload
with st.container():
    st.markdown("### 📁 3. Upload Training Data")
    uploaded_csv = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV should have two columns: 'prompt'/'response' or 'instruction'/'output'. Each row is a training example. You can format your data however you like (including with instruction templates already applied). If no file is uploaded, the default finetuning.csv will be used."
    )

    # Use default file if no upload
    if uploaded_csv is None:
        if os.path.exists("finetuning.csv"):
            st.info(
                "📄 Using default finetuning.csv file. Upload a different file to override.")
            # Preview default CSV
            df = pd.read_csv("finetuning.csv")
            st.dataframe(df.head(), width='stretch')
            st.caption(f"Total rows: {len(df)}")
        else:
            st.warning(
                "No CSV file uploaded and finetuning.csv not found. Please upload a CSV file.")
    else:
        # Preview uploaded CSV
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head(), width='stretch')
        st.caption(f"Total rows: {len(df)}")
    st.divider()

# Hyperparameters
with st.container():
    st.markdown("### 🎛️ 4. Fine-Tuning Hyperparameters")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Core Settings", "🎯 Optimization", "💾 Checkpointing"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=32, value=4,
                help="Number of samples per batch")
        with col2:
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3,
                                     help="Number of training epochs")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate", min_value=1e-6, max_value=1e-3, value=1e-5, format="%.6f",
                help="Initial learning rate (typically 10-100x lower than pre-training)")
        with col2:
            weight_decay = st.number_input(
                "Weight Decay", min_value=0.0, max_value=1.0, value=0.01, format="%.5f",
                help="L2 regularization strength")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            eval_interval = st.number_input(
                "Evaluation Interval", min_value=100, max_value=2000, value=500,
                help="Evaluate every N iterations")
        with col2:
            save_interval = st.number_input(
                "Save Interval", min_value=100, max_value=2000, value=500,
                help="Save checkpoint every N iterations")

    max_steps_per_epoch = st.number_input(
        "Max Steps per Epoch", min_value=100, max_value=5000, value=1000,
        help="Maximum number of training steps per epoch")

    max_length = st.number_input(
        "Max Sequence Length", min_value=128, max_value=2048, value=512,
        help="Maximum length for prompt+response sequences")

    # Quick stats
    _render_quick_stats(batch_size, learning_rate, epochs,
                        max_length, use_lora, lora_rank if use_lora else None)
    st.divider()

# Understand Your Model
with st.container():
    st.markdown("### 📚 5. Understand Your Model")

    # Show equations (after LoRA config so values are available)
    render_finetuning_equations(
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Show code implementation
    render_finetuning_code_snippets(use_lora=use_lora)
    st.divider()

# Start fine-tuning button
with st.container():
    st.markdown("### 🚀 6. Start Fine-Tuning")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col2:
        start_finetuning = st.button("🚀 Start Fine-Tuning", type="primary", width='stretch',
                                     help="Begin fine-tuning with current configuration")
    with col3:
        stop_training = st.button("⏹️ Stop Fine-Tuning", width='stretch',
                                  help="Stop the current fine-tuning run",
                                  disabled=not st.session_state.training_active)

    # Configuration summary before starting
    if start_finetuning:
        with st.expander("📋 Configuration Summary", expanded=True):
            config_summary = {
                "Checkpoint": selected_checkpoint["path"] if selected_checkpoint else "None",
                "Method": "LoRA" if use_lora else "Full Parameter",
                "Hyperparameters": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "max_steps_per_epoch": max_steps_per_epoch,
                    "max_length": max_length,
                    "eval_interval": eval_interval,
                    "save_interval": save_interval
                }
            }
            if use_lora:
                config_summary["LoRA"] = {
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "dropout": lora_dropout,
                    "target_modules": lora_target_modules
                }
            st.json(config_summary)
    st.divider()

if stop_training and st.session_state.training_active:
    with st.session_state.training_lock:
        if "training_active_flag" in st.session_state:
            st.session_state.training_active_flag[0] = False
    if "training_start_time" in st.session_state and "training_end_time" not in st.session_state:
        st.session_state.training_end_time = time.time()
    st.session_state.training_active = False
    st.rerun()

if start_finetuning:
    if not uploaded_csv and not os.path.exists("finetuning.csv"):
        st.error("Please upload a CSV file or ensure finetuning.csv exists.")
    elif st.session_state.training_active:
        st.warning("Training is already in progress!")
    else:
        _start_finetuning_workflow(
            selected_checkpoint["path"],
            uploaded_csv,
            batch_size,
            learning_rate,
            weight_decay,
            epochs,
            max_steps_per_epoch,
            eval_interval,
            save_interval,
            max_length,
            use_lora,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
        )

# Display training status
display_training_status(training_type="Fine-Tuning")
