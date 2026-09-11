import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from qrm.layers.attention import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
)
from qrm.layers.common import trunc_normal_init_
from qrm.layers.mlp import SwiGLU
from qrm.layers.normalization import rms_norm
from qrm.layers.sparse_embedding import CastedSparseEmbedding
from qrm.losses.trm import IGNORE_LABEL_ID, stablemax_cross_entropy
from qrm.models.trm.configuration_trm import TRMConfig

logger = logging.getLogger(__name__)


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor  # [B, seq_len+puzzle_emb_len, hidden_size]
    z_L: torch.Tensor  # [B, seq_len+puzzle_emb_len, hidden_size]


@dataclass
class TRMCarry:
    inner: TRMInnerCarry
    steps: torch.Tensor  # [B] int32
    halted: torch.Tensor  # [B] bool
    current_data: Optional[
        Dict[str, torch.Tensor]
    ]  # Current batch data being processed


@dataclass
class TRMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    q_halt_logits: Optional[torch.Tensor] = None
    q_continue_logits: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    # Return tensor instead of bool to avoid torch.compile Graph Break from .item()
    all_finish: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, torch.Tensor]] = None  # Evaluation metrics


class TRMBlock(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        if config.mlp_t:
            # MLP-T mode
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + config.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            # Standard Attention
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal,
            )

        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            # MLP-T: transpose before processing
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Standard Attention + Post-Norm
            hidden_states = rms_norm(
                hidden_states
                + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        # MLP + Post-Norm
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class ReasoningModule(nn.Module):

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.layers = nn.ModuleList([TRMBlock(config) for _ in range(config.L_layers)])

    def forward(self, hidden_states, injection, cos_sin):
        hidden_states = hidden_states + injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states


class TRMInner(nn.Module):

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Embedding layers
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = config.puzzle_emb_len
        if self.puzzle_emb_len == 0:
            self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)

        if config.puzzle_emb_ndim > 0:
            # CastedSparseEmbedding for puzzle embeddings
            # rank=None means full rank (no projection layer)
            puzzle_emb_rank = getattr(config, "puzzle_emb_rank", None)

            # Warn if low-rank is not beneficial (e.g., SHARED mode with 1 identifier)
            if puzzle_emb_rank is not None:
                if config.num_puzzle_identifiers <= puzzle_emb_rank:
                    logger.warning(
                        f"Low-rank embedding not beneficial: num_puzzle_identifiers="
                        f"{config.num_puzzle_identifiers} <= rank={puzzle_emb_rank}. "
                        f"Consider setting puzzle_emb_rank=null."
                    )
                elif config.num_puzzle_identifiers <= config.puzzle_emb_ndim:
                    # When num_identifiers <= dim, low-rank increases total params
                    logger.warning(
                        f"Low-rank embedding may increase params: num_puzzle_identifiers="
                        f"{config.num_puzzle_identifiers} <= puzzle_emb_ndim={config.puzzle_emb_ndim}. "
                        f"Full rank: {config.num_puzzle_identifiers * config.puzzle_emb_ndim}, "
                        f"Low-rank: {config.num_puzzle_identifiers * puzzle_emb_rank + puzzle_emb_rank * config.puzzle_emb_ndim}."
                    )

            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,  # Zero initialization
                cast_to=self.forward_dtype,
                rank=puzzle_emb_rank,
            )

        # Positional encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        # Reasoning layers
        self.L_level = ReasoningModule(config)

        # Initial state buffers
        # H_init/L_init: Initial hidden state vectors for recursive reasoning
        # persistent=True includes these in state_dict() for checkpointing
        # These are learned initial states that affect model behavior
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # FSQ layers
        if config.use_fsq:
            from qrm.layers.stochastic_fsq import StochasticFSQ
            self.fsq = StochasticFSQ(
                levels=config.fsq_levels,
                dim=config.hidden_size,
                top_k=config.fsq_top_k,
                temperature=config.fsq_temperature,
                pre_norm=config.fsq_pre_norm,
            )
            # FSQ residual weight: z_L = sigmoid(α) * z_L_quantized + (1 - sigmoid(α)) * z_L
            if config.fsq_residual_mode == "fixed":
                self.register_buffer(
                    "fsq_residual_weight",
                    torch.tensor(config.fsq_residual_weight),
                )
            elif config.fsq_residual_mode == "learned_scalar":
                # Store in logit space so sigmoid(init) ≈ fsq_residual_weight
                init_logit = torch.tensor(config.fsq_residual_weight).clamp(1e-6, 1 - 1e-6).logit()
                self.fsq_residual_weight = nn.Parameter(init_logit)
            elif config.fsq_residual_mode == "attention":
                from qrm.layers.fsq_attention import FSQAttention
                self.fsq_attention = FSQAttention(hidden_size=config.hidden_size)
        else:
            self.fsq = None

        # Q head special initialization
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input_ids, puzzle_identifiers):
        """Compute input embeddings."""
        embedding = self.embed_tokens(input_ids.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_emb = self.puzzle_emb(puzzle_identifiers)

            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size - puzzle_emb.shape[-1]
            )
            if pad_count > 0:
                puzzle_emb = F.pad(puzzle_emb, (0, pad_count))

            puzzle_emb = puzzle_emb.view(
                -1, self.puzzle_emb_len, self.config.hidden_size
            ).to(self.forward_dtype)
            embedding = torch.cat([puzzle_emb, embedding], dim=-2)

        # Learned positional encoding
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def _reasoning_cycles(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        input_embeddings: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        return_z_L_before_last_z_H: bool = False,
    ):
        """Execute H/L reasoning cycles.

        This method encapsulates the core reasoning loop, allowing QRM to
        inject FSQ between z_L and z_H in the last cycle.

        Args:
            z_H: [B, S, D] or [B, M, S, D] hidden state H
            z_L: [B, S, D] or [B, M, S, D] hidden state L
            input_embeddings: [B, S, D] input embeddings
            cos_sin: Rotary embeddings (optional)
            return_z_L_before_last_z_H: If True, return z_L after L_cycles
                but before the final z_H computation (for QRM FSQ injection)

        Returns:
            If return_z_L_before_last_z_H=False:
                (z_H, z_L) after all cycles
            If return_z_L_before_last_z_H=True:
                (z_H_before_last, z_L_after_L_cycles) - z_L ready for FSQ
        """
        # H_cycles-1 without grad
        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Last H_cycle: L_cycles iterations
        for _l in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

        # For QRM: return here before final z_H (FSQ will be injected)
        if return_z_L_before_last_z_H:
            return z_H, z_L

        # Final z_H computation
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        return z_H, z_L

    def forward(self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]):
        """TRMInner forward pass."""
        input_ids = batch["input_ids"]
        puzzle_identifiers = batch.get("puzzle_identifiers")

        # Input encoding
        input_embeddings = self._input_embeddings(input_ids, puzzle_identifiers)

        # Forward iterations
        z_H = carry.z_H
        z_L = carry.z_L

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # Execute reasoning cycles
        z_H, z_L = self._reasoning_cycles(
            z_H,
            z_L,
            input_embeddings,
            cos_sin,
            return_z_L_before_last_z_H=self.config.use_fsq,
        )

        z_continuous = None
        z_L_quantized_raw = None

        if self.config.use_fsq:
            # Save continuous z_L for reconstruction loss (before FSQ, detached)
            z_continuous = z_L.detach()

            # Determine sampling strategy based on training/inference mode
            # NOTE: do_sampling only has effect when fsq_top_k > 1.
            # With top_k=1, sampling and greedy produce identical results.
            do_sampling = (
                self.config.fsq_sampling_training
                if self.training
                else self.config.fsq_sampling_inference
            )
            return_top_k = self.config.fsq_residual_mode == "attention"
            z_L_quantized, _ = self.fsq(
                z_L,
                return_top_k=return_top_k,
                do_sampling=do_sampling,
            )

            if self.config.fsq_residual_mode == "fixed":
                B, M, S, D = z_L_quantized.shape
                assert M == 1, "M must be 1 for TRM when using FSQ"
                z_L_quantized = z_L_quantized.reshape(B * M, S, D)
                z_L_quantized_raw = z_L_quantized
                alpha = self.fsq_residual_weight
                z_L = alpha * z_L_quantized + (1 - alpha) * z_L
            elif self.config.fsq_residual_mode == "learned_scalar":
                B, M, S, D = z_L_quantized.shape
                assert M == 1, "M must be 1 for TRM when using FSQ"
                z_L_quantized = z_L_quantized.reshape(B * M, S, D)
                z_L_quantized_raw = z_L_quantized
                alpha = torch.sigmoid(self.fsq_residual_weight)
                z_L = alpha * z_L_quantized + (1 - alpha) * z_L
            elif self.config.fsq_residual_mode == "attention":
                # Cross-attention over top-k candidates
                z_L = self.fsq_attention(z_L, z_L_quantized)

            z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # LM Outputs
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # Detach to truncate computation graph for Truncated BPTT
        # Prevents gradients from flowing through carry to previous forward calls
        # Combined with torch.no_grad() for H_cycles-1 iterations:
        #   - Only last H_cycle has gradients
        #   - Carry passes across batches without gradient propagation
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return new_carry, logits, q_logits, z_continuous, z_L_quantized_raw

    def empty_carry(
        self, batch_size: int, device: torch.device = None
    ) -> TRMInnerCarry:
        """Create empty inner carry (uninitialized tensors).

        Note: Returns torch.empty, not initialized with H_init/L_init.
        Actual initialization happens in forward via reset_carry.
        """
        if device is None:
            device = self.H_init.device
        seq_len_with_puzzle = self.config.seq_len + self.puzzle_emb_len
        return TRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                seq_len_with_puzzle,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                seq_len_with_puzzle,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: TRMInnerCarry
    ) -> TRMInnerCarry:
        """Reset carry state for halted samples.

        Args:
            reset_flag: [B] bool tensor, True means needs reset.
            carry: Current inner carry.

        Returns:
            Reset inner carry.
        """
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )


class TRMForPuzzleSolving(PreTrainedModel):
    config_class = TRMConfig

    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.config = config
        self.model = TRMInner(config)

        # Carry runtime state (not saved)
        self._carry: Optional[TRMCarry] = None

        self.post_init()

    @property
    def puzzle_emb(self):
        """Convenience accessor for puzzle_emb, used by SignSGD optimizer."""
        return self.model.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        """Create initial carry.

        Key design:
        1. inner_carry uses empty_carry (uninitialized tensors), not H_init/L_init directly
        2. halted initialized to True, so first forward triggers reset_carry for initialization
        3. current_data initialized as empty_like, not None
        """
        # Get correct key (compatible with input_ids and inputs)
        if "input_ids" in batch:
            batch_size = batch["input_ids"].shape[0]
            device = batch["input_ids"].device
        else:
            batch_size = batch["inputs"].shape[0]
            device = batch["inputs"].device

        return TRMCarry(
            # Use empty_carry, actual initialization in forward via reset_carry
            inner=self.model.empty_carry(batch_size, device=device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            # Create empty_like dict, not None
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def reset_carry(self):
        """Reset carry."""
        self._carry = None

    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        puzzle_identifiers: torch.Tensor = None,
        carry: TRMCarry = None,
        batch: Dict[str, torch.Tensor] = None,
        return_keys: set = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # Support both input methods
        if batch is None:
            batch = {
                "input_ids": input_ids,
                "labels": labels,
                "puzzle_identifiers": puzzle_identifiers,
            }
            batch = {k: v for k, v in batch.items() if v is not None}

        if carry is None:
            carry = self._carry

        # Check if carry batch size matches current batch
        # If mismatch (e.g., epoch boundary, eval switch), need to reinitialize
        if "input_ids" in batch:
            current_batch_size = batch["input_ids"].shape[0]
        else:
            current_batch_size = batch["inputs"].shape[0]

        if carry is not None and carry.halted.shape[0] != current_batch_size:
            rank = int(os.environ.get("RANK", 0))
            logger.warning(
                f"[Rank {rank}] Carry batch size mismatch: "
                f"carry={carry.halted.shape[0]}, batch={current_batch_size}. "
                f"Reinitializing carry."
            )
            carry = None

        if carry is None:
            carry = self.initial_carry(batch)

        # Reset halted samples' state using TRMInner.reset_carry
        inner_carry = self.model.reset_carry(carry.halted, carry.inner)

        # Selective data update
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # For each key: if halted=True, use batch[k]; otherwise keep carry.current_data[k]
        # initial_carry created empty_like current_data, so no None check needed
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Inner forward
        inner_carry, logits, q_logits, z_continuous, z_L_quantized = self.model(inner_carry, new_current_data)

        q_halt_logits = q_logits[:, 0]
        q_continue_logits = q_logits[:, 1]

        # Update halt state
        target_q_continue = None
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                # Halt signal
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Target Q for continue (only when no_ACT_continue=False)
                if not self.config.no_ACT_continue:
                    _, _, next_q_logits, _, _ = self.model(inner_carry, new_current_data)
                    next_q_halt_logits = next_q_logits[:, 0]
                    next_q_continue_logits = next_q_logits[:, 1]
                    target_q_continue = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        # Update carry
        new_carry = TRMCarry(
            inner=inner_carry,  # inner_carry already detached in TRMInner.forward
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )
        self._carry = new_carry

        # Return tensor instead of bool to avoid torch.compile Graph Break from .item()
        # Single-element tensor can be used directly in if statements
        all_finish = halted.all()

        # Compute loss and metrics
        loss = None
        metrics = None
        lm_loss = None
        q_halt_loss = None
        q_continue_loss = None

        if "labels" in new_current_data and new_current_data["labels"] is not None:
            current_labels = new_current_data["labels"]
            mask = current_labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(
                -1
            )  # Avoid division by zero

            # Preds
            preds = logits.argmax(-1)

            # Correctness
            is_correct = mask & (preds == current_labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics - only count halted=True samples with valid labels
            valid_metrics = halted & (loss_counts > 0)

            with torch.no_grad():
                metrics = {
                    # Count of valid samples
                    "count": valid_metrics.sum(),
                    # Token-level accuracy (normalized by loss_divisor)
                    # Note: loss_divisor is [B, 1] for proper broadcasting [B, seq_len] / [B, 1]
                    "accuracy": torch.where(
                        valid_metrics,
                        (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                        torch.zeros_like(valid_metrics, dtype=torch.float32),
                    ).sum(),
                    # Sequence-level accuracy
                    "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                    # Q_halt prediction accuracy
                    "q_halt_accuracy": (
                        valid_metrics & ((q_halt_logits >= 0) == seq_is_correct)
                    ).sum(),
                    # Reasoning steps
                    "steps": torch.where(
                        valid_metrics, new_steps, torch.zeros_like(new_steps)
                    ).sum(),
                }

            # LM Loss
            per_sample_lm_loss = (
                stablemax_cross_entropy(logits, current_labels, valid_mask=mask)
                / loss_divisor
            ).sum(-1)  # [B]

            if self.config.weighted_em_loss:
                em_w = self.config.weighted_em_weight
                weight = torch.where(seq_is_correct, 1.0, 1.0 + em_w)  # [B]
                lm_loss = (weight * per_sample_lm_loss).sum()
            else:
                lm_loss = per_sample_lm_loss.sum()

            # Q-Halt Loss
            q_halt_loss = F.binary_cross_entropy_with_logits(
                q_halt_logits, seq_is_correct.to(q_halt_logits.dtype), reduction="sum"
            )

            # Q-Continue Loss (only when no_ACT_continue=False and target_q_continue exists)
            q_continue_loss = 0
            if target_q_continue is not None:
                q_continue_loss = F.binary_cross_entropy_with_logits(
                    q_continue_logits, target_q_continue, reduction="sum"
                )

            # Reconstruction loss (FSQ regularization)
            recon_loss = torch.tensor(0.0, device=logits.device)
            if (
                self.config.use_fsq
                and self.config.fsq_recon_weight > 0
                and z_continuous is not None
                and z_L_quantized is not None
                and self.config.fsq_residual_mode != "attention"
            ):
                recon_loss = F.mse_loss(z_L_quantized, z_continuous)

            # Update loss items in metrics
            with torch.no_grad():
                metrics["lm_loss"] = lm_loss.detach()
                if self.config.weighted_em_loss:
                    metrics["lm_unweighted_loss"] = per_sample_lm_loss.sum().detach()
                metrics["q_halt_loss"] = q_halt_loss.detach()
                if target_q_continue is not None:
                    metrics["q_continue_loss"] = (
                        q_continue_loss.detach()
                        if isinstance(q_continue_loss, torch.Tensor)
                        else torch.tensor(q_continue_loss)
                    )
                if self.config.use_fsq and self.config.fsq_recon_weight > 0:
                    metrics["recon_loss"] = recon_loss.detach()

            # Total loss
            fsq_recon_weight = self.config.fsq_recon_weight or 0
            loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + fsq_recon_weight * recon_loss

        if not return_dict:
            return new_carry, loss, {"logits": logits}, all_finish

        return TRMOutput(
            loss=loss,
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            halted=halted,
            all_finish=all_finish,
            metrics=metrics,
        )
