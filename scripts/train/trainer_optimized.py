from __future__ import annotations

# Original: scripts/train/trainer_optimized.py (renamed from trainer_new.py) from ace-step and woct0rdho fork
# Upstream repos:
#   - ACE-Step: https://github.com/ace-step/ACE-Step (repo HEAD: 6ae0852b1388de6dc0cca26b31a86d711f723cb3)
#   - woct0rdho/ACE-Step: https://github.com/woct0rdho/ACE-Step (repo HEAD: ea2ec6ba68be9c73254c2ec8f89a6965c6e5c3e8)
# Modified for: ACE-DATA_v2
# Modified by: methmx83 (ACE-DATA_v2)
# Modification date: 2025-08-31
# Modifications: Extensive refactor and optimisations for consumer GPU (quantisation hints, save callback, reduced VRAM footprint). See `third_party/THIRD_PARTY_LICENSES.md` for provenance and file-level notes.

import torch
import torch.nn.functional as F
import torch.utils.data
import warnings
import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)

# 1) Tensor-Core-Pfade für FP32 erlauben (TF32) + Matmul-Precision setzen
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Optional: Sichtprüfung
print(f"ℹ️ Matmul precision at import: {torch.get_float32_matmul_precision()}")

# 2) Lightning-Info-Banner auf WARNING drosseln
for name in ("pytorch_lightning", "lightning"):
    logging.getLogger(name).setLevel(logging.WARNING)

import warnings
warnings.filterwarnings(
    "ignore",
    message="Precision bf16-mixed is not supported by the model summary",
    category=UserWarning,
)

import argparse
import json
import os
import random
import shutil
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import hdf5plugin  # noqa: F401  # Needed for HDF5 compression plugins
import numpy as np
import safetensors.torch   

from natsort import natsorted
from peft import LoraConfig
from prodigyopt import Prodigy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

try:
    # 8‑bit optimiser and quantisation config are optional.  They
    # significantly reduce optimiser state memory【544999232868152†L154-L160】.
    import bitsandbytes as bnb  # type: ignore
except ImportError:
    bnb = None  # bitsandbytes is only required when using the 8‑bit optimiser

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class SaveLoRACallback(Callback):
    """Custom callback to save LoRA adapters without creating checkpoint files."""
    
    def __init__(self, save_every_n_steps, save_last, ckpt_path, adapter_name):
        self.save_every_n_steps = save_every_n_steps
        self.save_last = save_last
        self.ckpt_path = ckpt_path
        self.adapter_name = adapter_name

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.save_every_n_steps == 0:
            checkpoint_dir = self.ckpt_path or os.path.join(os.getcwd(), "./data/lora/r256_8bit")
            os.makedirs(checkpoint_dir, exist_ok=True)

            epoch = trainer.current_epoch
            step = trainer.global_step
            lora_name = f"epoch={epoch}-step={step}_lora"
            lora_path = os.path.join(checkpoint_dir, lora_name)
            os.makedirs(lora_path, exist_ok=True)

            pl_module.transformer.save_lora_adapter(lora_path, adapter_name=self.adapter_name)

            # Keep only latest checkpoints
            lora_paths = glob(os.path.join(checkpoint_dir, "*_lora"))
            lora_paths = natsorted(lora_paths)
            if len(lora_paths) > self.save_last:
                shutil.rmtree(lora_paths[0])


def set_seed(seed: int) -> None:
    """Set seeds for deterministic behaviour across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_tags(
    text_token_ids: torch.Tensor,
    mask: torch.Tensor,
    shuffle: bool,
    dropout: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shuffle and drop tags separated by commas.

    This function splits the tokenised prompt into tags separated by comma
    tokens, optionally shuffles them and applies dropout to drop some tags.
    The final token (BOS) is preserved.  The same logic as in the original
    script is retained.

    Args:
        text_token_ids: Long tensor of token ids including a BOS at the end.
        mask: Attention mask corresponding to `text_token_ids`.
        shuffle: If True, shuffle the order of tags.
        dropout: Probability of dropping individual tags.

    Returns:
        The augmented `text_token_ids` and its attention mask.
    """
    if not shuffle and not dropout:
        return text_token_ids, mask

    COMMA = 275  # token id for comma in ACE‑Step tokenizer
    bos = text_token_ids[-1:]
    text_token_ids = text_token_ids[:-1]

    # Split into tags based on commas
    tags: List[torch.Tensor] = []
    start_idx = 0
    _len = len(text_token_ids)
    for idx in range(_len):
        if text_token_ids[idx] == COMMA:
            if start_idx < idx:
                tags.append(text_token_ids[start_idx:idx])
            start_idx = idx + 1
    if start_idx < _len:
        tags.append(text_token_ids[start_idx:_len])

    # Shuffle with torch's RNG to respect PyTorch's global seed
    if shuffle:
        perm = torch.randperm(len(tags))
        tags = [tags[i] for i in perm]

    # Apply dropout to each tag
    if dropout:
        tags = [x for x in tags if torch.rand(()) > dropout]

    comma = torch.tensor([COMMA], dtype=text_token_ids.dtype)
    tags_and_commas: List[torch.Tensor] = []
    for x in tags:
        tags_and_commas.append(x)
        tags_and_commas.append(comma)
    if tags_and_commas:
        tags_and_commas[-1] = bos
    else:
        tags_and_commas.append(bos)

    text_token_ids = torch.cat(tags_and_commas)
    mask = mask[: len(text_token_ids)]
    return text_token_ids, mask


def pytree_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively cast tensors in a nested data structure to `dtype`."""
    if isinstance(x, list):
        return [pytree_to_dtype(y, dtype) for y in x]
    elif isinstance(x, dict):
        return {k: pytree_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
        return x.to(dtype)
    else:
        return x


class HDF5Dataset(Dataset):
    """Lazy dataset for per‑sample HDF5 files.

    This dataset reads a separate HDF5 file per audio sample.  Each file must
    contain the keys produced by the ACE‑Step preprocessing script.  To
    reduce I/O overhead, the dataset keeps no persistent file handles and
    converts arrays to PyTorch tensors on the fly.  Any exceptions during
    reading are caught to avoid crashing the training loop.
    """

    def __init__(
        self,
        dataset_path: str,
        dtype: torch.dtype,
        tag_shuffle: bool,
        tag_dropout: float,
    ) -> None:
        self.dataset_path = dataset_path
        self.dtype = dtype
        self.tag_shuffle = tag_shuffle
        self.tag_dropout = tag_dropout
        self.filenames = sorted(os.listdir(dataset_path))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = os.path.join(self.dataset_path, self.filenames[idx])
        try:
            with h5py.File(file_path, "r") as f:
                sample = {
                    k: torch.from_numpy(np.asarray(f[k]))
                    for k in f.keys()
                    if k != "keys"
                }
        except Exception as e:
            # Log and skip corrupted files rather than crashing
            print(f"Warning: ❌ failed to load {file_path}: {e}")
            # Return a dummy all‑zero sample to keep DataLoader length consistent
            # Note: downstream code should handle zero masks gracefully
            return {k: torch.zeros(1) for k in [
                "target_latents", "attention_mask", "text_token_ids",
                "text_attention_mask", "speaker_embds", "lyric_token_ids",
                "lyric_mask", "mert_ssl_hidden_states", "mhubert_ssl_hidden_states",
            ]}
        # Augment tags
        sample["text_token_ids"], sample["text_attention_mask"] = augment_tags(
            sample["text_token_ids"],
            sample["text_attention_mask"],
            self.tag_shuffle,
            self.tag_dropout,
        )
        sample["text_attention_mask"] = sample["text_attention_mask"].float()
        # Cast floating tensors to desired dtype (bf16 or fp16)
        sample = pytree_to_dtype(sample, self.dtype)
        return sample


class Pipeline(LightningModule):
    """Lightning wrapper around ACE‑Step for LoRA training."""

    def __init__(
        self,
        # Model
        checkpoint_dir: Optional[str] = None,
        T: int = 1000,
        shift: float = 3.0,
        timestep_densities_type: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        lora_config_path: Optional[str] = "./config/lora",
        last_lora_path: Optional[str] = None,
        # Data
        dataset_path: str = "./data/data_sets/train_set",
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        tag_dropout: float = 0.5,
        speaker_dropout: float = 0.0,
        lyrics_dropout: float = 0.0,
        # Optimiser
        ssl_coeff: float = 1.0,
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 1e-2,
        max_steps: int = 1000,
        warmup_steps: int = 10,
        # Others
        adapter_name: str = "lora_adapter",
        save_last: int = 5,
        precision: str = "bf16",
        compile_mode: Optional[str] = None,
        text_encoder_device: str = "cuda",
        ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Determine compute dtype based on precision
        if precision.startswith("bf16") and torch.cuda.is_bf16_supported():
            self.to_dtype = torch.bfloat16
        else:
            # Fall back to fp16 on GPUs that do not support bf16
            self.to_dtype = torch.float16
        self.to_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialise scheduler
        self.scheduler = self.get_scheduler()

        # Load model components
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)

        # Transformer (diffusion model)
        self.transformer = acestep_pipeline.ace_step_transformer.to(
            self.to_device, self.to_dtype
        )
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        # Enable gradient checkpointing to reduce memory footprint
        self.transformer.enable_gradient_checkpointing()

        # Text encoder – optionally run on CPU to save VRAM
        if text_encoder_device == "cpu":
            encoder_device = torch.device("cpu")
            encoder_dtype = torch.float32  # Use full precision on CPU
        else:
            encoder_device = self.to_device
            encoder_dtype = self.to_dtype
        self.text_encoder_model = acestep_pipeline.text_encoder_model.to(
            encoder_device, encoder_dtype
        )
        self.text_encoder_model.eval()
        self.text_encoder_model.requires_grad_(False)

        # Drop reference to pipeline to free memory
        del acestep_pipeline

        # Load LoRA configuration
        if not lora_config_path:
            raise ValueError("Please provide a --lora_config_path")
        with open(lora_config_path, encoding="utf-8") as f:
            lora_config_dict = json.load(f)
        lora_config = LoraConfig(**lora_config_dict)
        self.transformer.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)

        # Load previous LoRA weights if provided
        if last_lora_path:
            state_dict = safetensors.torch.load_file(last_lora_path)
            print("[DEBUG] LoRA keys loaded:", list(state_dict.keys())[:5])
            state_dict = {
                  k.replace(".weight", f".{adapter_name}.weight"): v
                  for k, v in state_dict.items()
            }
            self.transformer.load_state_dict(state_dict, strict=False)
            
            
        # Optionally compile the transformer for faster execution
        if compile_mode:
            try:
                # Only compile repeated blocks; dynamic=True allows varying input sizes
                self.transformer = torch.compile(
                    self.transformer,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
                print(f"Compiled transformer with torch.compile (mode={compile_mode})")
            except Exception as e:
                pass

        # Cast attention projection modules to compile individually.  This was
        # present in the original script; we keep it for continuity but allow
        # compilation to be skipped if torch.compile is unavailable.
        for module in self.transformer.projectors:
            try:
                module.forward = torch.compile(module.forward, dynamic=True)
            except Exception:
                pass
        try:
            self.transformer.encode = torch.compile(self.transformer.encode, dynamic=True)
            self.text_encoder_model = torch.compile(self.text_encoder_model, dynamic=True)
        except Exception:
            # If compilation fails, silently fall back to eager mode
            pass

    def get_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.hparams.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        # Collect trainable parameters (LoRA weights are the only trainable ones)
        trainable_params = [p for _, p in self.transformer.named_parameters() if p.requires_grad]

        opt_name = self.hparams.optimizer.lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
            lr_multiplier_final = 0.0
        elif opt_name == "adamw8bit":
            if bnb is None:
                raise ImportError(
                    "bitsandbytes not installed – install with `pip install bitsandbytes` to use the 8‑bit optimiser"
                )
            optimizer = bnb.optim.AdamW8bit(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
            lr_multiplier_final = 0.0
        elif opt_name == "prodigy":
            # See ShakkerAI guide: Prodigy adapts learning rates dynamically and is
            # effective for fine‑tuning tasks like DreamBooth LoRA training【143449840302135†screenshot】.
            if self.hparams.learning_rate < 0.1:
                print(
                    "Warning: Prodigy typically uses lr=1; adjust your --learning_rate if necessary."
                )
            if self.hparams.warmup_steps > 0:
                print(
                    "Warning: Prodigy often does not require warmup; consider setting --warmup_steps=0."
                )
            optimizer = Prodigy(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
                use_bias_correlection=True,
                safeguard_warmup=True,
                slice_p=11,
            )
            lr_multiplier_final = 0.1
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Set up linear warmup followed by linear decay.  Final value of LR can be
        # controlled via lr_multiplier_final for optimisers like Prodigy.
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                multiplier = max(0.0, 1.0 - progress)
                multiplier = multiplier * (1 - lr_multiplier_final) + lr_multiplier_final
                return multiplier

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        ds = HDF5Dataset(
            dataset_path=self.hparams.dataset_path,
            dtype=self.to_dtype,
            tag_shuffle=True,
            tag_dropout=self.hparams.tag_dropout,
        )
        loader = DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor,
        )
        return loader

    # Helpers to get sigmas and timesteps (identical to original)
    def get_sd3_sigmas(self, timesteps: torch.Tensor, device: torch.device, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz: int, device: torch.device) -> torch.Tensor:
        if self.hparams.timestep_densities_type == "logit_normal":
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(indices, 0, self.scheduler.config.num_train_timesteps - 1)
            timesteps = self.scheduler.timesteps[indices].to(device)
        else:
            raise ValueError(f"Unknown timestep_densities_type: {self.hparams.timestep_densities_type}")
        return timesteps

    def run_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        target_latents = batch["target_latents"]
        attention_mask = batch["attention_mask"]
        text_token_ids = batch["text_token_ids"]
        text_attention_mask = batch["text_attention_mask"]
        speaker_embds = batch["speaker_embds"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_mask"]
        mert_ssl_hidden_states = batch["mert_ssl_hidden_states"]
        mhubert_ssl_hidden_states = batch["mhubert_ssl_hidden_states"]

        device = self.to_device
        dtype = self.to_dtype

        # Forward pass through text encoder.  Move embeddings to GPU only if encoder runs on CPU.
        with torch.no_grad():
            outputs = self.text_encoder_model(
                input_ids=text_token_ids.to(self.text_encoder_model.device),
                attention_mask=text_attention_mask.to(self.text_encoder_model.device),
            )
            encoder_text_hidden_states = outputs.last_hidden_state.to(device, dtype)

        # Optional dropout for speaker and lyrics encodings
        if self.hparams.speaker_dropout and torch.rand(()) < self.hparams.speaker_dropout:
            speaker_embds = torch.zeros_like(speaker_embds)
        if self.hparams.lyrics_dropout and torch.rand(()) < self.hparams.lyrics_dropout:
            lyric_token_ids = torch.zeros_like(lyric_token_ids)
            lyric_mask = torch.zeros_like(lyric_mask)

        # Step 1: generate random noise and sample timesteps
        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise using flow matching schedule
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_latents.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_latents
        target = target_latents

        # Gather SSL hidden states if present
        all_ssl_hiden_states: List[torch.Tensor] = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # Predict noise
        transformer_output = self.transformer(
            hidden_states=noisy_image,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device, dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Precondition model outputs【401938786717703†L117-L133】
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss on valid timesteps
        mask = attention_mask[:, None, None, :]
        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()
        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"
        self.log(f"{prefix}/denoising_loss", loss, on_step=True, on_epoch=False)

        total_proj_loss = torch.tensor(0.0, device=device)
        for k, v in proj_losses:
            self.log(f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False)
            total_proj_loss += v
        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)
        if self.hparams.ssl_coeff:
            loss = loss + total_proj_loss * self.hparams.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False)

        # Log learning rate
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(f"{prefix}/learning_rate", learning_rate, on_step=True, on_epoch=False)
        # Log Prodigy specific internal state
        if self.hparams.optimizer.lower() == "prodigy":
            prodigy_d = self.optimizers().param_groups[0].get("d", None)
            if prodigy_d is not None:
                self.log(f"{prefix}/prodigy_d", prodigy_d, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.run_step(batch, batch_idx)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    # Model and scheduler parameters
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory with pre‑trained ACE‑Step weights")
    parser.add_argument("--shift", type=float, default=3.0, help="Shift for flow matching scheduler")
    parser.add_argument("--lora_config_path", type=str, default="./config/lora_config_transformer_only.json", help="JSON config for LoRA")
    parser.add_argument("--last_lora_path", type=str, default=None, help="Path to previously saved LoRA weights to resume training")
    parser.add_argument("--ckpt_path", type=str, default="./data/lora/r256_8bit", help="Lora Checkpoints")
    # Data
    parser.add_argument("--dataset_path", type=str, default="./data/data_sets/train_set", help="Path to preprocessed dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader worker processes")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory in DataLoader for faster host‑to‑GPU copies")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")
    parser.add_argument("--tag_dropout", type=float, default=0.5, help="Dropout probability for prompt tags")
    parser.add_argument("--speaker_dropout", type=float, default=0.0, help="Probability to drop speaker embeddings")
    parser.add_argument("--lyrics_dropout", type=float, default=0.0, help="Probability to drop lyric tokens")
    # Optimiser
    parser.add_argument("--ssl_coeff", type=float, default=1.0, help="Coefficient for SSL projection losses")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "prodigy"], help="Which optimiser to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam/Prodigy")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for Adam/Prodigy")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=-1, help="Maximum number of epochs (ignored if max_steps > 0)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Number of training steps to run")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_clip_val", type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm", help="Gradient clipping algorithm")
    # Misc
    parser.add_argument("--exp_name", type=str, default="ace-step_lora", help="Experiment name (used as adapter_name)")
    parser.add_argument("--precision", type=str, default="bf16", help="Mixed precision (bf16-mixed or 16-mixed)")
    parser.add_argument("--save_every_n_train_steps", type=int, default=100, help="Interval in steps to save LoRA weights")
    parser.add_argument("--save_last", type=int, default=5, help="Number of most recent LoRAs to keep on disk")
    parser.add_argument("--compile_mode", type=str, default=None, choices=["default", "reduce-overhead", "max-autotune", "none"])
    parser.add_argument("--text_encoder_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the text encoder on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")

    args = parser.parse_args(argv)
    if args.ckpt_path:
        args.ckpt_path = os.path.normpath(args.ckpt_path)
    # Set seeds
    set_seed(args.seed)

    model = Pipeline(
        checkpoint_dir=args.checkpoint_dir,
        shift=args.shift,
        lora_config_path=args.lora_config_path,
        last_lora_path=args.last_lora_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        tag_dropout=args.tag_dropout,
        speaker_dropout=args.speaker_dropout,
        lyrics_dropout=args.lyrics_dropout,
        ssl_coeff=args.ssl_coeff,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        adapter_name=args.exp_name,
        save_last=args.save_last,
        precision=args.precision,
        compile_mode=args.compile_mode,
        text_encoder_device=args.text_encoder_device,
        ckpt_path=args.ckpt_path,
    )
    
    # Create custom callback for saving LoRA adapters
    lora_callback = SaveLoRACallback(
        save_every_n_steps=args.save_every_n_train_steps,
        save_last=args.save_last,
        ckpt_path=args.ckpt_path,
        adapter_name=args.exp_name
    )
    
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=args.precision,
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,  # Disable internal checkpointing
        callbacks=[lora_callback],
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()