import logging
import re
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch import nn

from qrm.optim.sparse_embedding import DistributedCastedSparseEmbeddingSignSGD
from qrm.trainers.trm import TRMTrainer

logger = logging.getLogger(__name__)

# Pattern to detect depth-bucketed metric keys like "accuracy_d1_8", "reward_d16"
_BUCKET_SUFFIX_RE = re.compile(r"^(.+)_(d\d+(?:_\d+)?)$")


class QRMTrainer(TRMTrainer):
    """QRM Trainer, extends TRMTrainer.

    Core differences:
    1. create_optimizer: puzzle_emb path is model.trm_inner.puzzle_emb (QRM composition)
    2. training_step: supports gradient_accumulation_steps > 1
       (gradient accumulation = num_iterations MCTS iterations per optimizer update)
    3. log: depth-bucketed metrics divided by per-bucket counts
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Staging buffer: accumulates metrics within one MCTS search.
        # Flushed to _accumulated_metrics only when all_finish fires,
        # so partial searches at epoch boundaries are not logged.
        self._staged_metrics: Dict[str, float] = {}
        self._staged_count: int = 0
        self._staged_batch_size: int = 0

    def create_optimizer(self):
        """Create dual optimizers with QRM puzzle_emb path.

        QRM uses composition (QRMInner wraps TRMInner), so puzzle_emb path differs:
        - TRM: model.model.puzzle_emb
        - QRM: model.model.trm_inner.puzzle_emb

        QRMForPuzzleSolving provides a property shortcut, but create_optimizer
        needs to access the actual module for buffer iteration.
        """
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()

        # Initialize EMA (same as TRM)
        if self.use_ema and self.ema_helper is None:
            from qrm.trainers.trm import EMAHelper

            self.ema_helper = EMAHelper(mu=self.ema_rate)
            self.ema_helper.register(self.model)

        # Main optimizer: AdamAtan2 (same as TRM)
        if self.optimizer is None:
            from adam_atan2_pytorch import AdamAtan2

            self.optimizer = AdamAtan2(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
            )

        # SignSGD optimizer: for puzzle_emb (QRM-specific path)
        if self.signsgd_optimizer is None:
            inner_model = self.model
            if hasattr(inner_model, "_orig_mod"):
                inner_model = inner_model._orig_mod

            # QRM path: model.model.trm_inner.puzzle_emb
            puzzle_emb = None
            if hasattr(inner_model, "model"):
                model_inner = inner_model.model
                if hasattr(model_inner, "trm_inner") and hasattr(
                    model_inner.trm_inner, "puzzle_emb"
                ):
                    puzzle_emb = model_inner.trm_inner.puzzle_emb
                elif hasattr(model_inner, "puzzle_emb"):
                    # Fallback to TRM path
                    puzzle_emb = model_inner.puzzle_emb

            if puzzle_emb is not None:
                self.signsgd_optimizer = DistributedCastedSparseEmbeddingSignSGD(
                    list(puzzle_emb.buffers()),
                    lr=0,
                    weight_decay=self.puzzle_emb_weight_decay,
                    world_size=world_size,
                )

        return self.optimizer

    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log to handle depth-bucketed metrics.

        Depth-bucketed metrics (e.g. accuracy_d1_8) are divided by their
        own count (count_d1_8), not the global count.  max_depth is divided
        by count_final (only populated on the last MCTS iteration).
        """
        if self._metrics_count > 0 and "loss" in logs:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            # Distributed reduce (SUM) — same as TRM
            if world_size > 1:
                metric_keys = list(self._accumulated_metrics.keys())
                metric_values = torch.tensor(
                    [self._accumulated_metrics[k] for k in metric_keys],
                    dtype=torch.float32,
                    device="cuda",
                )
                dist.reduce(metric_values, dst=0)
                for i, k in enumerate(metric_keys):
                    self._accumulated_metrics[k] = metric_values[i].item()

            if world_size > 1:
                batch_size_tensor = torch.tensor(
                    [self._accumulated_batch_size], dtype=torch.float32, device="cuda"
                )
                dist.reduce(batch_size_tensor, dst=0)
                total_samples = int(batch_size_tensor.item())
            else:
                total_samples = self._accumulated_batch_size

            if self.is_world_process_zero():
                am = self._accumulated_metrics
                count = max(am.get("count", 1), 1)

                # Collect all count_* keys for bucket-specific division
                bucket_counts = {}
                for k, v in am.items():
                    if k.startswith("count_"):
                        bucket_counts[k[6:]] = max(v, 1)  # "count_d1_8" -> "d1_8"

                for k, v in am.items():
                    if k.startswith("count"):
                        continue  # counts are divisors, don't log them

                    m = _BUCKET_SUFFIX_RE.match(k)
                    if m:
                        # Bucketed metric: divide by its bucket count
                        suffix = m.group(2)
                        divisor = bucket_counts.get(suffix, count)
                        logs[f"train/{k}"] = v / divisor
                    elif k == "max_depth":
                        divisor = max(am.get("count_final", 1), 1)
                        logs[f"train/{k}"] = v / divisor
                    elif k.endswith("loss"):
                        logs[f"train/{k}"] = v / total_samples
                    else:
                        logs[f"train/{k}"] = v / count

            self._accumulated_metrics = {}
            self._metrics_count = 0
            self._accumulated_batch_size = 0

        # SignSGD lr
        if self.signsgd_optimizer is not None and "learning_rate" in logs:
            logs["learning_rate_signsgd"] = self.signsgd_optimizer.param_groups[0]["lr"]
            logs["learning_rate_main"] = logs.pop("learning_rate")

        # FSQ residual weight
        inner = self.model
        if hasattr(inner, "module"):
            inner = inner.module
        if hasattr(inner, "_orig_mod"):
            inner = inner._orig_mod
        if getattr(inner.config, "fsq_residual_mode", None) == "learned_scalar":
            model_inner = inner.model if hasattr(inner, "model") else inner
            if hasattr(model_inner, "trm_inner"):
                model_inner = model_inner.trm_inner
            if hasattr(model_inner, "fsq_residual_weight"):
                logs["train/fsq_residual_alpha"] = torch.sigmoid(
                    model_inner.fsq_residual_weight
                ).item()

        # Skip TRMTrainer.log (which would re-process _accumulated_metrics)
        # Call Trainer.log directly
        from transformers import Trainer
        Trainer.log(self, logs, start_time)

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Training step with gradient accumulation support.

        Each training_step call is one MCTS iteration inside the model.
        gradient_accumulation_steps = num_iterations means one optimizer
        update per full tree search.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # Loss scaling: divide by local_batch_size (same as TRM),
        # then by gradient_accumulation_steps so HF Trainer's summed loss
        # reflects the per-iteration average (not the raw sum over iterations).
        local_batch_size = inputs["input_ids"].shape[0]
        scaled_loss = loss / (local_batch_size * self.args.gradient_accumulation_steps)

        self.accelerator.backward(scaled_loss)

        # Accumulate training metrics into a staging buffer.
        # Only flush to the main accumulator when a full search completes
        # (all_finish), so partial searches at epoch boundaries don't
        # pollute the logged averages.
        if hasattr(outputs, "metrics") and outputs.metrics is not None:
            for k, v in outputs.metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                self._staged_metrics[k] = self._staged_metrics.get(k, 0.0) + val
            self._staged_count += 1
            self._staged_batch_size += local_batch_size

        if hasattr(outputs, "all_finish") and outputs.all_finish:
            # Full search complete — flush staged metrics to main accumulator
            for k, v in self._staged_metrics.items():
                self._accumulated_metrics[k] = (
                    self._accumulated_metrics.get(k, 0.0) + v
                )
            self._metrics_count += self._staged_count
            self._accumulated_batch_size += self._staged_batch_size
            self._staged_metrics = {}
            self._staged_count = 0
            self._staged_batch_size = 0

        return scaled_loss.detach()
