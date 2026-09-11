"""TRM Trainer - Trainer implementation for Tiny Recursive Models.

Key features:
1. Dual optimizer architecture: AdamAtan2 (main params) + SignSGD (puzzle_emb)
2. Custom cosine schedule with warmup, supporting lr_min_ratio
3. Both optimizers use the same schedule curve but different base_lr
4. SignSGD called immediately after main optimizer.step()
5. EMA (Exponential Moving Average) support
"""

import copy
import logging
import math
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from adam_atan2_pytorch import AdamAtan2
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput

from qrm.optim.sparse_embedding import DistributedCastedSparseEmbeddingSignSGD

_TRANSFORMERS_VERSION = version.parse(transformers.__version__)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def _clean_param_name(name: str) -> str:
    """Remove '_orig_mod.' prefix added by torch.compile."""
    return name.replace("_orig_mod.", "")


def log_weight_comparison(
    params_a: Dict,
    params_b: Dict,
    label_a: str = "Model A",
    label_b: str = "Model B",
    prefix: str = "[Weight]",
):
    """Log weight comparison between two parameter sets for debugging."""
    sample_param_name = list(params_a.keys())[0]

    mean_a = params_a[sample_param_name].data.mean().item()
    std_a = params_a[sample_param_name].data.std().item()

    mean_b = params_b[sample_param_name].data.mean().item()
    std_b = params_b[sample_param_name].data.std().item()

    logger.info(f"{prefix} Weight comparison for '{sample_param_name}':")
    logger.info(f"{prefix}   {label_a}: mean={mean_a:.6f}, std={std_a:.6f}")
    logger.info(f"{prefix}   {label_b}: mean={mean_b:.6f}, std={std_b:.6f}")


# =============================================================================
# EMA Helper
# =============================================================================


class EMAHelper:
    """Exponential Moving Average helper for smoothing model parameters.

    EMA formula: shadow = mu * shadow + (1 - mu) * param

    Usage:
    1. Call register(model) at training start to initialize shadow weights
    2. Call update(model) after each training step to update shadow
    3. Use ema_copy(model) during evaluation to get EMA version of the model

    Note: torch.compile adds '_orig_mod.' prefix to param names, handled during matching.
    """

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}
        self._swapped = False  # Tracks if shadow is in swapped state (holding train weights instead of EMA)

    def register(self, module: nn.Module):
        """Register shadow copy of model parameters."""
        # Handle DDP / DataParallel
        if hasattr(module, "module"):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                clean_name = _clean_param_name(name)
                self.shadow[clean_name] = param.data.clone()

    def update(self, module: nn.Module):
        """Update shadow weights: shadow = mu * shadow + (1 - mu) * param.

        Raises:
            RuntimeError: If called while in swapped state (shadow holds train weights).
        """
        if self._swapped:
            raise RuntimeError(
                "[EMA] Cannot update EMA while in swapped state! "
                "shadow currently holds training weights, not EMA weights. "
                "Call swap_weights() to restore before updating."
            )

        if hasattr(module, "module"):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                clean_name = _clean_param_name(name)
                self.shadow[clean_name].data = (
                    self.mu * self.shadow[clean_name].data
                    + (1.0 - self.mu) * param.data
                )

    def ema(self, module: nn.Module):
        """Replace model parameters with shadow weights."""
        if hasattr(module, "module"):
            module = module.module

        total_params = 0
        replaced_count = 0
        has_orig_mod = False
        for name, param in module.named_parameters():
            clean_name = _clean_param_name(name)
            if name != clean_name:
                has_orig_mod = True
            if param.requires_grad:
                total_params += 1
                if clean_name in self.shadow:
                    param.data.copy_(self.shadow[clean_name].data)
                    replaced_count += 1

        if has_orig_mod:
            logger.info(
                f"[EMA] ema: compiled model detected (_orig_mod prefix), replaced {replaced_count}/{total_params} params"
            )
        else:
            logger.info(f"[EMA] ema: replaced {replaced_count}/{total_params} params")

    def swap_weights(self, module: nn.Module):
        """Swap model weights with shadow weights in-place.

        Used to temporarily apply EMA weights during eval while preserving torch.compile optimization.
        Call twice to restore original state.

        State transitions:
        - _swapped=False → _swapped=True: Model becomes EMA weights, shadow holds train weights
        - _swapped=True → _swapped=False: Model restored to train weights, shadow restored to EMA

        Args:
            module: Model whose weights will be swapped.
        """
        if hasattr(module, "module"):
            module = module.module

        total_params = 0
        swap_count = 0
        has_orig_mod = False
        for name, param in module.named_parameters():
            clean_name = _clean_param_name(name)
            if name != clean_name:
                has_orig_mod = True
            if param.requires_grad:
                total_params += 1
                if clean_name in self.shadow:
                    # Swap param.data and shadow[clean_name].data
                    tmp = param.data.clone()
                    param.data.copy_(self.shadow[clean_name].data)
                    self.shadow[clean_name].data.copy_(tmp)
                    swap_count += 1

        # Toggle swap state
        self._swapped = not self._swapped
        state_str = (
            "swapped (model=EMA, shadow=Train)"
            if self._swapped
            else "restored (model=Train, shadow=EMA)"
        )

        if has_orig_mod:
            logger.info(
                f"[EMA] swap_weights: compiled model detected (_orig_mod prefix), swapped {swap_count}/{total_params} params, state={state_str}"
            )
        else:
            logger.info(
                f"[EMA] swap_weights: swapped {swap_count}/{total_params} params, state={state_str}"
            )

    def ema_copy(self, module: nn.Module) -> nn.Module:
        """Return a deep copy of the model using EMA weights."""
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """Return state dictionary for serialization."""
        # Move to CPU for space efficiency and compatibility
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()}, "mu": self.mu}

    def load_state_dict(self, state_dict, device=None):
        """Load state dictionary.

        Args:
            state_dict: State dictionary to load.
            device: Target device. If None, keeps original device.
        """
        self.shadow = state_dict["shadow"]
        self.mu = state_dict.get("mu", self.mu)

        # Move shadow to specified device if provided
        if device is not None:
            self.shadow = {k: v.to(device) for k, v in self.shadow.items()}


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def cosine_schedule_with_warmup_lr(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
) -> float:
    """Cosine schedule with warmup.

    Differences from HuggingFace's get_cosine_schedule_with_warmup:
    1. Supports min_ratio parameter (minimum learning rate ratio)
    2. Returns actual learning rate value, not a scaling factor

    Args:
        current_step: Current training step.
        base_lr: Base learning rate.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_ratio: Minimum learning rate ratio, default 0.0.
        num_cycles: Number of cosine cycles, default 0.5.

    Returns:
        Learning rate for the current step.
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


class DualOptimizerCallback(TrainerCallback):
    """Dual optimizer callback - manages main optimizer lr and SignSGD optimizer calls.

    Implements dual optimizer training loop:
    - on_pre_optimizer_step: Set main optimizer lr before optimizer.step()
    - on_optimizer_step: Call SignSGD optimizer after main optimizer.step()
    """

    def __init__(self, trainer: "TRMTrainer"):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Set main optimizer lr before optimizer.step()."""
        trainer = self.trainer

        # Calculate main optimizer lr
        # Note: state.global_step hasn't been incremented yet, so we use +1
        main_lr = cosine_schedule_with_warmup_lr(
            current_step=state.global_step + 1,
            base_lr=args.learning_rate,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=trainer._num_training_steps,
            min_ratio=trainer.lr_min_ratio,
        )

        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = main_lr

        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Call SignSGD optimizer after main optimizer.step().

        Call timing in HF Trainer:
        1. self.optimizer.step()       # Already executed
        2. on_optimizer_step callback  # <- Current position
        3. lr_scheduler.step()         # Not yet executed
        4. model.zero_grad()           # Not yet executed (only clears Parameter grads, not Buffer)
        """
        trainer = self.trainer

        if trainer.signsgd_optimizer is not None:
            # Calculate SignSGD lr (same schedule, different base_lr)
            signsgd_lr = cosine_schedule_with_warmup_lr(
                current_step=state.global_step + 1,
                base_lr=trainer.puzzle_emb_lr,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=trainer._num_training_steps,
                min_ratio=trainer.lr_min_ratio,
            )

            for param_group in trainer.signsgd_optimizer.param_groups:
                param_group["lr"] = signsgd_lr

            trainer.signsgd_optimizer.step()
            trainer.signsgd_optimizer.zero_grad()  # Buffer grads need manual zeroing

        return control


class EMACallback(TrainerCallback):
    """EMA Callback - updates EMA weights after each training step."""

    def __init__(self, trainer: "TRMTrainer"):
        self.trainer = trainer

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Update EMA after each training step."""
        if self.trainer.ema_helper is not None:
            self.trainer.ema_helper.update(self.trainer.model)
        return control


# =============================================================================
# TRM Trainer
# =============================================================================


class TRMTrainer(Trainer):
    """TRM Trainer - Trainer implementation for Tiny Recursive Models.

    Key differences from HuggingFace Trainer:
    1. Dual optimizer: AdamAtan2 (main params) + SignSGD (puzzle_emb buffers)
    2. Custom learning rate schedule: supports lr_min_ratio
    3. Loss scaling: loss / global_batch_size
    4. Multi-step inference until halt during evaluation
    5. EMA (Exponential Moving Average) support
    """

    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        train_dataset=None,
        eval_dataset=None,
        # TRM-specific parameters
        puzzle_emb_lr: float = None,
        puzzle_emb_weight_decay: float = None,
        lr_min_ratio: float = 0.0,
        # EMA parameters
        ema: bool = False,
        ema_rate: float = 0.999,
        # Whether to use compiled model during eval (via swap_weights instead of deep copy)
        eval_use_compile: bool = False,
        **kwargs,
    ):
        # Extract user callbacks to add together with internal callbacks
        user_callbacks = kwargs.pop("callbacks", None) or []

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=user_callbacks,
            **kwargs,
        )

        # TRM-specific configuration
        self.puzzle_emb_lr = (
            puzzle_emb_lr if puzzle_emb_lr is not None else args.learning_rate
        )
        self.puzzle_emb_weight_decay = (
            puzzle_emb_weight_decay
            if puzzle_emb_weight_decay is not None
            else args.weight_decay
        )
        self.lr_min_ratio = lr_min_ratio

        # EMA configuration
        self.use_ema = ema
        self.ema_rate = ema_rate
        self.ema_helper: Optional[EMAHelper] = None
        self.eval_use_compile = eval_use_compile

        # SignSGD optimizer (created in create_optimizer)
        self.signsgd_optimizer = None

        # Training steps (set in create_scheduler)
        self._num_training_steps = None

        # Add dual optimizer callback
        self.add_callback(DualOptimizerCallback(self))

        # Add EMA callback (only when enabled)
        if self.use_ema:
            self.add_callback(EMACallback(self))

        # Training metrics accumulation
        self._accumulated_metrics: Dict[str, float] = {}
        self._metrics_count: int = 0
        self._accumulated_batch_size: int = (
            0  # Accumulated actual batch size (since per_device_train_batch_size=1)
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save checkpoint - extended to save additional state.

        Version changes:
        - transformers < 4.47.0: _save_checkpoint(self, model, trial, metrics=None)
        - transformers >= 4.47.0: _save_checkpoint(self, model, trial)
        """
        if _TRANSFORMERS_VERSION < version.parse("4.47.0"):
            super()._save_checkpoint(model, trial, metrics=metrics)
        else:
            super()._save_checkpoint(model, trial)

        if self.args.should_save:
            checkpoint_folder = (
                f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
            )
            self._save_trm_extra_state(checkpoint_folder)

    def _save_trm_extra_state(self, checkpoint_folder: str):
        """Save TRM-specific additional state (SignSGD, EMA, Dataset)."""
        # Save dataset state
        # Note: When dataloader_num_workers > 0, main process's train_dataset._iters is not updated,
        # so we compute the correct _iters value from global_step
        if hasattr(self.train_dataset, "_iters") and self.args.eval_steps:
            computed_iters = self.state.global_step // self.args.eval_steps
            dataset_state = {"_iters": computed_iters}
            dataset_state_path = os.path.join(checkpoint_folder, "dataset_state.pt")
            torch.save(dataset_state, dataset_state_path)
            logger.info(
                f"[TRM] Saved dataset state: {dataset_state} (global_step={self.state.global_step}, eval_steps={self.args.eval_steps})"
            )

        # Save SignSGD optimizer state
        if self.signsgd_optimizer is not None:
            signsgd_state_path = os.path.join(checkpoint_folder, "signsgd_optimizer.pt")
            torch.save(self.signsgd_optimizer.state_dict(), signsgd_state_path)

        # Save EMA state
        if self.ema_helper is not None:
            ema_state_path = os.path.join(checkpoint_folder, "ema_state.pt")
            torch.save(self.ema_helper.state_dict(), ema_state_path)

    def save_final_checkpoint(self, output_dir: str):
        """Save complete final checkpoint (model, EMA, SignSGD, Dataset state).

        Unlike save_model(), this saves complete training state for resume or loading EMA weights.

        Args:
            output_dir: Directory to save checkpoint.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights
        self.save_model(output_dir=output_dir)

        # Save trainer state (includes global_step, etc.)
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Save TRM additional state
        if self.args.should_save:
            self._save_trm_extra_state(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from checkpoint - extended to restore additional state.

        Note: This method is only called in FSDP/SageMaker mode.
        Regular DDP mode uses _load_optimizer_and_scheduler.
        """
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._load_trm_extra_state(resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Load optimizer and scheduler state - extended to restore TRM additional state.

        This method is called in all modes (including regular DDP).
        """
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None:
            self._load_trm_extra_state(checkpoint)

    def _load_trm_extra_state(self, resume_from_checkpoint):
        """Load TRM-specific additional state (SignSGD, EMA, Dataset)."""
        logger.info(f"[TRM] Loading extra state from {resume_from_checkpoint}")

        # Restore dataset state
        # Note: When dataloader_num_workers > 0, dataset runs in worker processes,
        # so main process's _iters is not updated. We compute _iters from checkpoint's global_step.
        # Formula: _iters = global_step // eval_steps (each iter = eval_steps steps)
        if hasattr(self.train_dataset, "_iters") and self.args.eval_steps:
            # Parse global_step from checkpoint path (format: .../checkpoint-{step})
            checkpoint_name = os.path.basename(resume_from_checkpoint.rstrip("/"))
            if checkpoint_name.startswith("checkpoint-"):
                try:
                    resumed_global_step = int(checkpoint_name.split("-")[1])
                    computed_iters = resumed_global_step // self.args.eval_steps
                    self.train_dataset._iters = computed_iters
                    logger.info(
                        f"[TRM] Computed dataset _iters={computed_iters} from global_step={resumed_global_step}, eval_steps={self.args.eval_steps}"
                    )
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"[TRM] Failed to parse global_step from checkpoint path: {resume_from_checkpoint}, error: {e}"
                    )

        # Restore SignSGD optimizer state
        signsgd_state_path = os.path.join(
            resume_from_checkpoint, "signsgd_optimizer.pt"
        )
        if os.path.exists(signsgd_state_path) and self.signsgd_optimizer is not None:
            signsgd_state = torch.load(signsgd_state_path, map_location="cpu")
            self.signsgd_optimizer.load_state_dict(signsgd_state)
            # load_state_dict overwrites param_group defaults (including
            # world_size and weight_decay) with the values saved at checkpoint
            # time. Resuming on a different nproc must use the *current*
            # world_size, or the all_gather buffer is sized wrong (crash) or
            # skipped entirely (silently desynced puzzle embeddings across
            # ranks). Similarly, puzzle_emb_weight_decay must reflect the
            # current run's value so CLI overrides at resume take effect.
            current_world_size = dist.get_world_size() if dist.is_initialized() else 1
            for pg in self.signsgd_optimizer.param_groups:
                pg["world_size"] = current_world_size
                pg["weight_decay"] = self.puzzle_emb_weight_decay
            logger.info(
                f"[TRM] Loaded SignSGD optimizer state from {signsgd_state_path} "
                f"(world_size overridden to {current_world_size}, "
                f"weight_decay overridden to {self.puzzle_emb_weight_decay})"
            )
        else:
            logger.info(
                f"[TRM] SignSGD state not found or optimizer is None (path={signsgd_state_path}, optimizer={self.signsgd_optimizer is not None})"
            )

        # Restore EMA state
        ema_state_path = os.path.join(resume_from_checkpoint, "ema_state.pt")
        if os.path.exists(ema_state_path) and self.use_ema:
            if self.ema_helper is None:
                self.ema_helper = EMAHelper(mu=self.ema_rate)
            ema_state = torch.load(ema_state_path, map_location="cpu")
            device = next(self.model.parameters()).device
            self.ema_helper.load_state_dict(ema_state, device=device)
            logger.info(
                f"[TRM] Loaded EMA state from {ema_state_path}, mu={self.ema_helper.mu}, num_shadow_params={len(self.ema_helper.shadow)}"
            )
        else:
            logger.info(
                f"[TRM] EMA state not loaded (path_exists={os.path.exists(ema_state_path)}, use_ema={self.use_ema})"
            )

    def create_optimizer(self):
        """Create dual optimizers.

        Original logic:
        - Both optimizers start with lr=0, set by callback at each step
        - AdamAtan2 manages model.parameters()
        - SignSGD manages puzzle_emb.buffers()

        Note: beta1/beta2 use TrainingArguments' adam_beta1/adam_beta2.
        TRM defaults: beta1=0.9, beta2=0.95; HF defaults: beta1=0.9, beta2=0.999.

        DeepSpeed compatibility:
        - If DeepSpeed config doesn't define optimizer, DeepSpeed calls this method to create AdamAtan2
        - If DeepSpeed config defines optimizer, it uses DeepSpeed's optimizer (not recommended)
        - To ensure AdamAtan2 is used, don't define "optimizer" field in DeepSpeed config
        """
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()

        # Initialize EMA
        # Done in create_optimizer because model is already on correct device at this point
        if self.use_ema and self.ema_helper is None:
            self.ema_helper = EMAHelper(mu=self.ema_rate)
            self.ema_helper.register(self.model)

        # Main optimizer: AdamAtan2
        # Uses args.adam_beta1, args.adam_beta2 (corresponding to cfg_pretrain.yaml's beta1, beta2)
        # Check self.optimizer is None to avoid duplicate creation (may be called twice in DeepSpeed mode)
        # Note: adam_atan2_pytorch requires lr > 0, so we use learning_rate as placeholder
        # Actual lr will be overwritten by DualOptimizerCallback.on_pre_optimizer_step
        if self.optimizer is None:
            self.optimizer = AdamAtan2(
                self.model.parameters(),
                lr=self.args.learning_rate,  # Placeholder (adam_atan2 requires lr > 0), callback sets actual value
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
            )

        # SignSGD optimizer: for puzzle_emb
        # Handle compiled model structure
        # Note: SignSGD is independent of main optimizer, always needs to be created
        if self.signsgd_optimizer is None:
            inner_model = self.model
            if hasattr(inner_model, "_orig_mod"):
                inner_model = inner_model._orig_mod

            if hasattr(inner_model, "model") and hasattr(
                inner_model.model, "puzzle_emb"
            ):
                self.signsgd_optimizer = DistributedCastedSparseEmbeddingSignSGD(
                    list(inner_model.model.puzzle_emb.buffers()),
                    lr=0,  # Initial lr=0, set by callback
                    weight_decay=self.puzzle_emb_weight_decay,
                    world_size=world_size,
                )

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create learning rate scheduler - uses placeholder scheduler.

        TRM uses custom cosine_schedule_with_warmup_lr.
        We set lr manually in DualOptimizerCallback, so this is a placeholder.

        Note: HF Trainer calls lr_scheduler.step() after optimizer.step().
        Placeholder scheduler doesn't change anything.
        """
        from torch.optim.lr_scheduler import LambdaLR

        # Save training steps for callback use
        self._num_training_steps = num_training_steps

        # Placeholder scheduler - returns fixed 1.0, doesn't affect lr
        # Actual lr is set by DualOptimizerCallback at each step
        self.lr_scheduler = LambdaLR(optimizer or self.optimizer, lambda _: 1.0)

        return self.lr_scheduler

    def _get_learning_rate(self):
        """Override HF Trainer's lr getter.

        Read actual lr from optimizer param_groups, not from placeholder scheduler.
        """
        return self.optimizer.param_groups[0]["lr"]

    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log method to add SignSGD lr and training metrics."""
        # Add accumulated training metrics
        # Only add for training logs (check for "loss" key)
        if self._metrics_count > 0 and "loss" in logs:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            # Distributed reduce (SUM)
            if world_size > 1:
                metric_keys = list(self._accumulated_metrics.keys())
                metric_values = torch.tensor(
                    [self._accumulated_metrics[k] for k in metric_keys],
                    dtype=torch.float32,
                    device="cuda",
                )
                dist.reduce(metric_values, dst=0)

                # Update with reduced values
                for i, k in enumerate(metric_keys):
                    self._accumulated_metrics[k] = metric_values[i].item()

            # Distributed reduce batch_size (SUM)
            if world_size > 1:
                batch_size_tensor = torch.tensor(
                    [self._accumulated_batch_size], dtype=torch.float32, device="cuda"
                )
                dist.reduce(batch_size_tensor, dst=0)
                total_samples = int(batch_size_tensor.item())
            else:
                total_samples = self._accumulated_batch_size

            # Only add to logs on rank 0
            if self.is_world_process_zero():
                # Global count after reduce
                count = max(self._accumulated_metrics.get("count", 1), 1)

                for k, v in self._accumulated_metrics.items():
                    # Loss items divide by total_samples, others divide by count
                    if k.endswith("loss"):
                        logs[f"train/{k}"] = v / total_samples
                    else:
                        logs[f"train/{k}"] = v / count

            # Reset accumulators
            self._accumulated_metrics = {}
            self._metrics_count = 0
            self._accumulated_batch_size = 0

        # Add SignSGD lr
        if self.signsgd_optimizer is not None and "learning_rate" in logs:
            logs["learning_rate_signsgd"] = self.signsgd_optimizer.param_groups[0]["lr"]
            # Rename main lr to distinguish
            logs["learning_rate_main"] = logs.pop("learning_rate")

        # Log FSQ residual weight (learned_scalar mode)
        inner = self.model
        if hasattr(inner, "module"):  # DDP
            inner = inner.module
        if hasattr(inner, "_orig_mod"):  # torch.compile
            inner = inner._orig_mod
        if getattr(inner.config, "use_fsq", False) and inner.config.fsq_residual_mode == "learned_scalar":
            # inner is TRMForPuzzleSolving; fsq_residual_weight lives on inner.model (TRMInner)
            logs["train/fsq_residual_alpha"] = torch.sigmoid(inner.model.fsq_residual_weight).item()

        super().log(logs, start_time)

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Training step.

        Key points:
        1. Loss scaling: loss / global_batch_size
        2. lr setting handled by DualOptimizerCallback
        3. metrics accumulation

        Version changes:
        - transformers < 4.46.0: training_step(self, model, inputs)
        - transformers >= 4.46.0: training_step(self, model, inputs, num_items_in_batch=None)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            # Use return_outputs=True to get full outputs (including metrics)
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # Loss scaling
        # Original TRM: ((1 / global_batch_size) * loss).backward() + all_reduce(grad) [SUM]
        # HF Trainer (DDP): backward() + all_reduce(grad) / world_size [MEAN]
        #
        # To be equivalent to TRM, we only divide by local_batch_size:
        # grad = SUM(local_grad / local_batch_size) / world_size
        #      = SUM(local_grad) / (local_batch_size * world_size)
        #      = SUM(local_grad) / global_batch_size  ✓
        local_batch_size = inputs["input_ids"].shape[0]
        scaled_loss = loss / local_batch_size

        if self.args.gradient_accumulation_steps > 1:
            # scaled_loss = scaled_loss / self.args.gradient_accumulation_steps
            raise NotImplementedError

        self.accelerator.backward(scaled_loss)

        # Accumulate training metrics
        if hasattr(outputs, "metrics") and outputs.metrics is not None:
            for k, v in outputs.metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                self._accumulated_metrics[k] = (
                    self._accumulated_metrics.get(k, 0.0) + val
                )
            self._metrics_count += 1
            self._accumulated_batch_size += (
                local_batch_size  # Accumulate actual batch size
            )

        return scaled_loss.detach()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Evaluation loop - multi-step inference until halt.

        Key features:
        1. Uses torch.inference_mode() for entire evaluation
        2. Reinitializes carry for each batch
        3. while True loop until all_finish
        4. If EMA enabled, uses EMA model for evaluation
        5. Uses model's returned metrics dict
        6. Distributed reduce metrics to rank 0
        7. Divide by count to get average
        """
        # Get distributed info
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # Use inference_mode for entire evaluation
        with torch.inference_mode():
            # Get original training model (only unwrap DDP, keep torch.compile optimization)
            # torch.compile wrapper proxies attribute access, can directly access initial_carry(), _carry, etc.
            # Note: Don't unwrap _orig_mod, otherwise evaluation uses uncompiled model, slower
            training_model = self.model
            if hasattr(training_model, "module"):
                training_model = training_model.module
            # Keep compiled model, don't unwrap _orig_mod

            # If EMA enabled, use EMA model for evaluation
            # Use swap_weights to temporarily replace weights, preserving torch.compile optimization
            need_restore_carry = False
            need_restore_weights = False
            saved_carry = None

            logger.info(
                f"[TRM Eval] use_ema={self.use_ema}, ema_helper={self.ema_helper is not None}, eval_use_compile={self.eval_use_compile}"
            )

            if self.use_ema and self.ema_helper is not None:
                if self.eval_use_compile:
                    # EMA mode + compile: use swap_weights to temporarily swap weights, keep compiled model
                    # 1. Save training carry
                    saved_carry = training_model._carry
                    need_restore_carry = True

                    # 2. Swap weights (original weights ↔ EMA weights)
                    self.ema_helper.swap_weights(training_model)
                    need_restore_weights = True

                    # 3. Use same compiled model (training_model now has EMA weights)
                    eval_model = training_model
                    logger.info(
                        f"[TRM Eval] Using EMA weights with compiled model (mu={self.ema_helper.mu})"
                    )

                    log_weight_comparison(
                        params_a={
                            _clean_param_name(n): p
                            for n, p in eval_model.named_parameters()
                        },
                        params_b=self.ema_helper.shadow,
                        label_a="Eval (EMA)",
                        label_b="Train (in shadow)",
                        prefix="[TRM Eval]",
                    )

                else:
                    # EMA mode + no compile: use deep copy (original logic)
                    eval_model = self.ema_helper.ema_copy(training_model)
                    logger.info(
                        f"[TRM Eval] Using EMA model (deep copy, no compile) (mu={self.ema_helper.mu})"
                    )

                    # Print weight comparison: eval_model (EMA weights) vs training_model (Train weights)
                    log_weight_comparison(
                        params_a={
                            _clean_param_name(n): p
                            for n, p in eval_model.named_parameters()
                        },
                        params_b={
                            _clean_param_name(n): p
                            for n, p in training_model.named_parameters()
                        },
                        label_a="Eval (EMA)",
                        label_b="Train",
                        prefix="[TRM Eval]",
                    )
            else:
                # Non-EMA mode: use training model, need to save/restore carry
                eval_model = training_model
                saved_carry = training_model._carry
                need_restore_carry = True
                logger.info("[TRM Eval] Using original model for evaluation (no EMA)")

            eval_model.eval()

            all_preds = []
            all_labels = []
            num_batches = 0

            # For aggregating model-returned metrics
            # metric_keys and metric_values initialized after first batch
            metric_keys = None
            metric_values = None

            # Show progress with tqdm
            # Note: IterableDataset DataLoader doesn't have __len__, need to get from dataset
            from tqdm import tqdm

            total_batches = None
            if hasattr(dataloader, "dataset") and hasattr(
                dataloader.dataset, "__len__"
            ):
                try:
                    total_batches = len(dataloader.dataset)
                except Exception:
                    pass
            dataloader_with_progress = tqdm(
                enumerate(dataloader),
                desc=description,
                total=total_batches,
                disable=not self.is_local_process_zero(),
            )

            for step, inputs in dataloader_with_progress:
                batch = self._prepare_inputs(inputs)

                # Explicitly create new carry for each batch
                carry = eval_model.initial_carry(batch)

                # Multi-step inference - while True loop until all_finish
                while True:
                    # Explicitly pass carry
                    outputs = eval_model(carry=carry, batch=batch, return_dict=True)
                    carry = eval_model._carry  # Get updated carry

                    # all_finish could be True, False, None (init)
                    if outputs.all_finish:
                        break

                num_batches += 1

                # Aggregate model-returned metrics
                # Only collect metrics at last step (all_finish=True)
                if outputs.metrics is not None:
                    batch_metrics = outputs.metrics

                    # First batch: initialize metric_keys and metric_values
                    if metric_keys is None:
                        # Sort keys to ensure consistent order across processes
                        metric_keys = list(sorted(batch_metrics.keys()))
                        metric_values = torch.zeros(
                            len(metric_keys),
                            dtype=torch.float32,
                            device=self.args.device,
                        )

                    # Accumulate metrics
                    for i, k in enumerate(metric_keys):
                        if k in batch_metrics:
                            v = batch_metrics[k]
                            metric_values[i] += (
                                v.float() if isinstance(v, torch.Tensor) else float(v)
                            )

                # Collect predictions and labels
                if outputs.logits is not None:
                    all_preds.append(outputs.logits.argmax(-1).cpu())

                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu())

            # Distributed reduce metrics to rank 0
            if metric_values is not None and world_size > 1:
                dist.reduce(metric_values, dst=0)

            # Compute final metrics on rank 0
            metrics = {}
            total_count = 0

            if metric_values is not None:
                # Convert to dict
                reduced_metrics = {
                    k: metric_values[i].item() for i, k in enumerate(metric_keys)
                }

                # Extract count and compute average
                total_count = reduced_metrics.pop("count", 1)
                total_count = max(total_count, 1)  # Avoid division by zero

                # All metrics divided by count
                for k, v in reduced_metrics.items():
                    metrics[f"{metric_key_prefix}_{k}"] = v / total_count

            # Add batch count for num_samples
            metrics[f"{metric_key_prefix}_num_batches"] = num_batches

            # Broadcast metrics to all processes
            # HF Trainer expects all processes to have same metrics (for logging/callbacks)
            if world_size > 1:
                # Broadcast metrics to all processes
                metrics_list = [metrics] if rank == 0 else [None]
                dist.broadcast_object_list(metrics_list, src=0)
                metrics = metrics_list[0]

                # Also broadcast total_count for num_samples
                total_count_tensor = torch.tensor(
                    [total_count], device=self.args.device
                )
                dist.broadcast(total_count_tensor, src=0)
                total_count = total_count_tensor.item()

            # Restore training state
            # Restore carry
            if need_restore_carry:
                training_model._carry = saved_carry

            # Restore weights (need to swap back in EMA mode)
            if need_restore_weights:
                self.ema_helper.swap_weights(training_model)

                log_weight_comparison(
                    params_a={
                        _clean_param_name(n): p
                        for n, p in training_model.named_parameters()
                    },
                    params_b=self.ema_helper.shadow,
                    label_a="Recovered Train",
                    label_b="EMA (in shadow)",
                    prefix="[TRM Eval]",
                )

        # Return results
        # Note: predictions and label_ids only contain local data, not aggregated
        # This is consistent with original TRM (each rank saves independently)
        return EvalLoopOutput(
            predictions=torch.cat(all_preds) if all_preds else None,
            label_ids=torch.cat(all_labels) if all_labels else None,
            metrics=metrics,
            num_samples=int(total_count),
        )
