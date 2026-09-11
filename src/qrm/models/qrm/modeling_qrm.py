"""QRM Model Implementation.

QRM (Quantized Recursive Model) extends TRM with:
1. Stochastic FSQ quantization after z_L
2. Multi-candidate training with weighted loss
3. MCTS tree search over FSQ candidates

Architecture:
- QRMInner: Core model with FSQ integration
- QRMForPuzzleSolving: Full model with loss computation and MCTS tree

Each forward() call = one MCTS iteration (select -> expand -> update tree).
Gradient accumulation spans num_iterations calls = one full tree search.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from qrm.layers.stochastic_fsq import StochasticFSQ
from qrm.losses import IGNORE_LABEL_ID, qrm_diversity_loss, qrm_tree_loss
from qrm.mcts import QRMMCTSManager
from qrm.models.qrm.configuration_qrm import QRMConfig
from qrm.models.trm.modeling_trm import TRMInner, TRMInnerCarry

logger = logging.getLogger(__name__)


@dataclass
class QRMInnerCarry:
    """Carry state for QRMInner.

    Same as TRMInnerCarry - FSQ does not affect carry state.
    """

    z_H: torch.Tensor  # [B, seq_len+puzzle_emb_len, hidden_size]
    z_L: torch.Tensor  # [B, seq_len+puzzle_emb_len, hidden_size]


@dataclass
class QRMOutput(ModelOutput):
    """Output of QRMForPuzzleSolving.

    Attributes:
        loss: Total loss (tree + recon + div)
        logits: [B, M, N, V] logits for all candidates
        z_continuous: [B, N, D] continuous latent before FSQ (for recon loss)
        z_quantized: [B, M, N, D] quantized latent after FSQ
        indices: [B, M, N] FSQ codebook indices
        weights: [B, M] per-candidate weights (reward-based)
        priors: [B, M] per-candidate priors from FSQ (for PUCT)
        metrics: Evaluation metrics dict
        all_finish: Tensor scalar bool - True when all samples finished
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    z_continuous: Optional[torch.Tensor] = None
    z_quantized: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, torch.Tensor]] = None
    all_finish: Optional[torch.Tensor] = None


class QRMInner(nn.Module):
    """QRM Inner model with FSQ integration.

    Extends TRMInner by adding StochasticFSQ between z_L and z_H in the
    last reasoning cycle. This creates M candidate paths that are all
    processed through the final z_H computation.

    Forward pass:
    1. Input embeddings (same as TRM)
    2. H_cycles-1 iterations + L_cycles of last H_cycle (get z_L before final z_H)
    3. FSQ quantization of z_L -> [B, M, S, D] candidates
    4. Expand z_H to candidate dimension and compute final z_H for all candidates
    5. LM head on quantized representations

    Carry state:
    - During forward: temporarily stores all M candidates (invalid state)
    - After external model selects best candidate: updated to valid [B, S, D] state

    Output shape convention:
    - Returns [B, M, N, V] logits for M candidates (M = top_k)
    """

    def __init__(self, config: QRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # === Reuse TRMInner components ===
        # We compose rather than inherit to avoid complexity
        self.trm_inner = TRMInner(config)

        # === FSQ Layer ===
        self.fsq = StochasticFSQ(
            levels=config.fsq_levels,
            dim=config.hidden_size,
            top_k=config.fsq_top_k,
            temperature=config.fsq_temperature,
            pre_norm=config.fsq_pre_norm,
        )

        # === FSQ Residual Weight ===
        # z_L_blended = α * z_L_quantized + (1-α) * z_L_continuous
        # Default α=1.0 = pure quantization (backward compat)
        if config.fsq_residual_mode == "fixed":
            self.register_buffer(
                "fsq_residual_weight",
                torch.tensor(config.fsq_residual_weight),
            )
        elif config.fsq_residual_mode == "learned_scalar":
            init_logit = torch.tensor(config.fsq_residual_weight).clamp(1e-6, 1 - 1e-6).logit()
            self.fsq_residual_weight = nn.Parameter(init_logit)

        # Separate LM head for QRM (operates on quantized representations)
        # Note: We create our own lm_head because output shape differs
        from qrm.layers.attention import CastedLinear

        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def puzzle_emb_len(self):
        return self.trm_inner.puzzle_emb_len

    def forward(
        self,
        carry: QRMInnerCarry,
        batch: Dict[str, torch.Tensor],
        return_top_k: bool = True,
        do_sampling: bool = False,
        need_priors: bool = False,
    ) -> Tuple[QRMInnerCarry, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """QRMInner forward pass.

        Args:
            carry: QRMInnerCarry with z_H and z_L [B, S, D]
            batch: Dict with input_ids, puzzle_identifiers, etc.
            return_top_k: If True, return all top-k candidates
            do_sampling: If True and return_top_k=False, sample from candidates
            need_priors: If True, compute PUCT priors from FSQ log-probs

        Returns:
            new_carry: Updated carry state [B, M, S, D] (INVALID - needs external selection)
            logits: [B, M, N, V] logits for candidates (N = seq_len, excludes puzzle_emb)
            z_continuous: [B, S, D] continuous z_L before FSQ (for recon loss)
            z_quantized: [B, M, S, D] quantized z_L after FSQ (S = seq_len + puzzle_emb_len)
            indices: [B, M, S] FSQ codebook indices
            priors: [B, M] per-candidate priors (None if need_priors=False)
        """
        input_ids = batch["input_ids"]
        puzzle_identifiers = batch.get("puzzle_identifiers")

        # Input encoding (reuse TRMInner's method)
        input_embeddings = self.trm_inner._input_embeddings(
            input_ids, puzzle_identifiers
        )

        # Get z_L before final z_H computation
        z_H = carry.z_H  # [B, S, D]
        z_L = carry.z_L  # [B, S, D]

        cos_sin = (
            self.trm_inner.rotary_emb()
            if hasattr(self.trm_inner, "rotary_emb")
            else None
        )

        # Execute reasoning cycles, stop before final z_H
        z_H, z_L = self.trm_inner._reasoning_cycles(
            z_H,
            z_L,
            input_embeddings,
            cos_sin,
            return_z_L_before_last_z_H=True,
        )

        # Save continuous z_L for reconstruction loss (before FSQ, detached)
        z_continuous = z_L.detach()  # [B, S, D]

        # FSQ quantization on entire z_L (including puzzle_emb positions)
        priors = None
        if need_priors:
            z_L_quantized, indices, diagnostics = self.fsq(
                z_L, return_top_k=True, return_diagnostics=True,
            )
            # Compute per-candidate priors from FSQ top-k log-probs
            # topk_log_probs: [B, N, M] -> mean across positions -> softmax
            topk_log_probs = diagnostics["topk_log_probs"]  # [B, N, M]
            candidate_log_probs = topk_log_probs.mean(dim=1)  # [B, M]
            priors = F.softmax(candidate_log_probs, dim=-1)  # [B, M]
        else:
            z_L_quantized, indices = self.fsq(
                z_L, return_top_k=return_top_k, do_sampling=do_sampling,
            )

        B, M, S, D = z_L_quantized.shape

        # Apply residual blending: z_L_blended = α * z_quantized + (1-α) * z_continuous
        if self.config.fsq_residual_mode == "fixed":
            alpha = self.fsq_residual_weight
        elif self.config.fsq_residual_mode == "learned_scalar":
            alpha = torch.sigmoid(self.fsq_residual_weight)

        z_L_for_z_H = z_L_quantized  # Default: pure quantization (α=1)
        # Static check avoids graph break when alpha is a tensor
        needs_blend = (
            self.config.fsq_residual_mode == "learned_scalar"
            or self.config.fsq_residual_weight < 1.0
        )
        if needs_blend:
            z_L_expanded = z_L.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, S, D]
            z_L_for_z_H = alpha * z_L_quantized + (1 - alpha) * z_L_expanded

        # Expand z_H to candidate dimension for final z_H computation
        z_H_expanded = z_H.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, S, D]

        # Compute final z_H for all M candidates in parallel
        z_H_flat = z_H_expanded.reshape(B * M, S, D)
        z_L_flat = z_L_for_z_H.reshape(B * M, S, D)

        z_H_final_flat = self.trm_inner.L_level(z_H_flat, z_L_flat, cos_sin=cos_sin)
        z_H_final = z_H_final_flat.reshape(B, M, S, D)

        # LM head: exclude puzzle_emb positions, then compute logits
        z_H_seq = z_H_final[:, :, self.puzzle_emb_len :]  # [B, M, N, D]
        N = z_H_seq.shape[2]

        z_flat = z_H_seq.reshape(B * M * N, D)
        logits_flat = self.lm_head(z_flat)
        logits = logits_flat.reshape(B, M, N, self.config.vocab_size)

        # Create carry with ALL candidates (INVALID state - [B, M, S, D])
        new_carry = QRMInnerCarry(
            z_H=z_H_final.detach(),
            z_L=z_L_for_z_H.detach(),
        )

        return new_carry, logits, z_continuous, z_L_quantized, indices, priors

    def empty_carry(
        self, batch_size: int, device: torch.device = None
    ) -> QRMInnerCarry:
        """Create empty carry (delegates to TRMInner)."""
        trm_carry = self.trm_inner.empty_carry(batch_size, device)
        return QRMInnerCarry(z_H=trm_carry.z_H, z_L=trm_carry.z_L)

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: QRMInnerCarry
    ) -> QRMInnerCarry:
        """Reset carry state for halted samples (delegates to TRMInner)."""
        trm_carry = TRMInnerCarry(z_H=carry.z_H, z_L=carry.z_L)
        reset_trm_carry = self.trm_inner.reset_carry(reset_flag, trm_carry)
        return QRMInnerCarry(z_H=reset_trm_carry.z_H, z_L=reset_trm_carry.z_L)

    @staticmethod
    def select_best_carry(
        carry: QRMInnerCarry,
        best_indices: torch.Tensor,
    ) -> QRMInnerCarry:
        """Select best candidate from multi-candidate carry.

        Converts INVALID [B, M, S, D] carry to VALID [B, S, D] carry.
        """
        B, M, S, D = carry.z_H.shape
        idx = best_indices.view(B, 1, 1, 1).expand(-1, -1, S, D)
        z_H_best = torch.gather(carry.z_H, dim=1, index=idx).squeeze(1)
        z_L_best = torch.gather(carry.z_L, dim=1, index=idx).squeeze(1)
        return QRMInnerCarry(z_H=z_H_best, z_L=z_L_best)


@dataclass
class QRMCarry:
    """Carry state for QRMForPuzzleSolving."""

    inner: QRMInnerCarry
    steps: torch.Tensor  # [B] int32 - current step count
    current_data: Optional[Dict[str, torch.Tensor]]  # Current batch data


class QRMForPuzzleSolving(PreTrainedModel):
    """QRM model for puzzle solving tasks.

    Each forward() call is one MCTS iteration:
    1. Select node from tree (or initialize tree on first call)
    2. Expand: run QRMInner to get M candidates
    3. Evaluate: compute loss, reward, weights
    4. Update tree with children, backpropagate

    Training: gradient_accumulation_steps = num_iterations forward calls per update
    Inference: call forward() in a loop until all_finish
    """

    config_class = QRMConfig

    def __init__(self, config: QRMConfig):
        super().__init__(config)
        self.config = config
        self.model = QRMInner(config)

        # Runtime state (not saved)
        self._carry: Optional[QRMCarry] = None
        self._mcts: Optional[QRMMCTSManager] = None

        self.post_init()

    @property
    def puzzle_emb(self):
        """Convenience accessor for puzzle_emb, used by SignSGD optimizer."""
        return self.model.trm_inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> QRMCarry:
        """Create initial carry."""
        if "input_ids" in batch:
            batch_size = batch["input_ids"].shape[0]
            device = batch["input_ids"].device
        else:
            batch_size = batch["inputs"].shape[0]
            device = batch["inputs"].device

        return QRMCarry(
            inner=self.model.empty_carry(batch_size, device=device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def reset_carry(self):
        """Reset carry and MCTS tree."""
        self._carry = None
        if self._mcts is not None:
            self._mcts.release()
        self._mcts = None

    def _compute_reward(
        self,
        logits: torch.Tensor,  # [B, M, N, V]
        labels: torch.Tensor,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-candidate reward, token accuracy, and exact match.

        Reward = em_weight * exact_match + token_acc_weight * token_accuracy

        Returns:
            rewards: [B, M] per-candidate rewards in [0, 1]
            token_acc: [B, M] per-candidate token accuracy
            exact_match: [B, M] per-candidate exact match (0 or 1)
        """
        B, M, N, V = logits.shape

        # Get predictions
        preds = logits.argmax(dim=-1)  # [B, M, N]

        # Expand labels: [B, N] -> [B, M, N]
        labels_expanded = labels.unsqueeze(1).expand(-1, M, -1)

        # Mask for valid positions
        mask = labels_expanded != IGNORE_LABEL_ID  # [B, M, N]
        valid_counts = mask.sum(dim=-1).clamp(min=1)  # [B, M]

        # Token-level accuracy
        correct = (preds == labels_expanded) & mask  # [B, M, N]
        token_acc = correct.sum(dim=-1).float() / valid_counts  # [B, M]

        # Exact match
        exact_match = (correct.sum(dim=-1) == valid_counts).float()  # [B, M]

        # Combined reward
        reward = (
            self.config.em_weight * exact_match
            + self.config.token_acc_weight * token_acc
        )
        return reward, token_acc, exact_match

    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        puzzle_identifiers: torch.Tensor = None,
        carry: QRMCarry = None,
        batch: Dict[str, torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> QRMOutput:
        """Forward pass = one MCTS iteration.

        Each call: select node -> expand (inner forward) -> compute loss -> update tree.
        """
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

        # Mid-search: ignore incoming batch entirely, use cached data.
        # HF Trainer sends different batches per gradient accumulation step,
        # but MCTS needs the same data for all iterations of one tree search.
        if carry is not None and (carry.steps > 0).all():
            B = carry.steps.shape[0]
            device = carry.steps.device
            inner_carry = carry.inner
            new_current_data = carry.current_data
        else:
            # New sequence (step 0) or first call: use the incoming batch
            if "input_ids" in batch:
                current_batch_size = batch["input_ids"].shape[0]
                device = batch["input_ids"].device
            else:
                current_batch_size = batch["inputs"].shape[0]
                device = batch["inputs"].device

            if carry is not None and carry.steps.shape[0] != current_batch_size:
                rank = int(os.environ.get("RANK", 0))
                logger.warning(
                    f"[Rank {rank}] Carry batch size mismatch: carry={carry.steps.shape[0]}, "
                    f"batch={current_batch_size}. Reinitializing carry."
                )
                carry = None
                if self._mcts is not None:
                    self._mcts.release()
                self._mcts = None

            if carry is None:
                carry = self.initial_carry(batch)

            # Determine if this is a new sequence (step 0)
            # NOTE: Should be 1 here?
            is_new_sequence = carry.steps == 0

            # Reset carry for new sequences
            inner_carry = self.model.reset_carry(is_new_sequence, carry.inner)

            new_current_data = {
                k: torch.where(
                    is_new_sequence.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
                )
                for k, v in carry.current_data.items()
            }
            B = current_batch_size

        if self.training:
            return self._forward_train(B, device, carry, inner_carry, new_current_data, return_dict)
        else:
            return self._forward_inference(B, device, carry, inner_carry, new_current_data, return_dict)

    def _forward_train(self, B, device, carry, inner_carry, new_current_data, return_dict):
        """Training: MCTS tree search. One call = one iteration."""

        # Initialize tree on first call
        if self._mcts is None:
            self._mcts = QRMMCTSManager(
                max_depth=self.config.halt_max_steps,
                search_rule=self.config.search_rule,
                c_uct=self.config.c_uct,
                c_puct=self.config.c_puct,
                terminal_selection_mode=self.config.terminal_selection_mode,
                q_normalize=self.config.q_normalize,
            )
            self._mcts.initialize_roots(inner_carry)

        # Select node to expand per batch element
        selections = []
        selected_zH = []
        selected_zL = []
        child_depths = []
        for b in range(B):
            sel = self._mcts.select_node(b)
            selections.append(sel)
            if sel is not None:
                selected_zH.append(sel.node.carry.z_H.squeeze(0))
                selected_zL.append(sel.node.carry.z_L.squeeze(0))
                child_depths.append(sel.node.depth + 1)
            else:
                selected_zH.append(self._mcts.roots[b].carry.z_H.squeeze(0))
                selected_zL.append(self._mcts.roots[b].carry.z_L.squeeze(0))
                child_depths.append(0)

        selected_carry = QRMInnerCarry(
            z_H=torch.stack(selected_zH),
            z_L=torch.stack(selected_zL),
        )
        child_depth_tensor = torch.tensor(
            child_depths, dtype=torch.float32, device=device,
        )

        # Inner forward (top-k for MCTS)
        need_priors = self.config.search_rule == "puct"
        inner_carry_out, logits, z_continuous, z_quantized, indices, priors = self.model(
            selected_carry, new_current_data,
            return_top_k=True, need_priors=need_priors,
        )

        new_steps = carry.steps + 1
        loss = None
        metrics = None
        weights = None
        best_idx = None

        _, M, N, V = logits.shape

        if "labels" in new_current_data and new_current_data["labels"] is not None:
            current_labels = new_current_data["labels"]

            with torch.no_grad():
                rewards, token_acc, exact_match_vals = self._compute_reward(
                    logits, current_labels,
                )
                best_idx = rewards.argmax(dim=1)  # [B]

            # Compute weights: w = (0.2 + 0.8 * reward) * (1 - alpha * d/D_tree)
            # Paper Eq. 23: depth penalty encourages efficient (shallow) reasoning paths
            # In Milestone 1 (single-step), depth d = new_steps, D_tree = halt_max_steps
            depth_ratio = child_depth_tensor.unsqueeze(1) / self.config.halt_max_steps
            depth_penalty = 1.0 - self.config.alpha * depth_ratio
            weights = (0.2 + 0.8 * rewards) * depth_penalty  # [B, M]

            # === Tree Loss ===
            tree_loss = qrm_tree_loss(logits, current_labels, weights)

            # === Reconstruction Loss ===
            # MSE between z_continuous (detached) and z_quantized (mean over M)
            # z_continuous: [B, S, D], z_quantized: [B, M, S, D] where S = seq_len + puzzle_emb_len
            recon_loss = torch.tensor(0.0, device=logits.device)
            if self.config.lambda_recon > 0:
                # Use mean of quantized candidates for reconstruction
                z_quantized_mean = z_quantized.mean(dim=1)  # [B, S, D]
                recon_loss = F.mse_loss(z_quantized_mean, z_continuous)

            # === Diversity Loss ===
            div_loss = torch.tensor(0.0, device=logits.device)
            if self.config.lambda_div > 0 and M > 1:
                div_loss = qrm_diversity_loss(
                    z_quantized,
                    beta=self.config.diversity_beta,
                    seq_aggregation=self.config.diversity_seq_aggregation,
                )

            loss = (
                self.config.lambda_tree * tree_loss
                + self.config.lambda_recon * recon_loss
                + self.config.lambda_div * div_loss
            )

            # Update tree with expansion results
            with torch.no_grad():
                for b in range(B):
                    if selections[b] is None:
                        continue
                    child_carry_b = QRMInnerCarry(
                        z_H=inner_carry_out.z_H[b : b + 1],
                        z_L=inner_carry_out.z_L[b : b + 1],
                    )
                    self._mcts.sync_children_from_expansion(
                        selections[b].node,
                        child_carry_b,
                        rewards=rewards[b].detach(),
                        priors=priors[b].detach() if priors is not None else None,
                        token_accuracy=token_acc[b].detach(),
                        exact_match=exact_match_vals[b].detach(),
                    )
                    # Propagate rewards of *all* M newly expanded children —
                    # each observed reward counts as one simulation.
                    self._mcts.backpropagate_all_children(selections[b].node)

            with torch.no_grad():
                best_preds = torch.gather(
                    logits.argmax(dim=-1),
                    dim=1,
                    index=best_idx.view(B, 1, 1).expand(-1, -1, N),
                ).squeeze(1)

                mask = current_labels != IGNORE_LABEL_ID
                valid_counts = mask.sum(dim=-1).clamp(min=1)
                correct = (best_preds == current_labels) & mask
                best_token_acc = correct.sum(dim=-1).float() / valid_counts
                best_exact_match = (correct.sum(dim=-1) == valid_counts).float()

                metrics = {
                    "count": torch.tensor(B, device=logits.device),
                    "accuracy": best_token_acc.sum(),
                    "exact_accuracy": best_exact_match.sum(),
                    "mean_reward": rewards.mean(dim=1).sum(),
                    "tree_loss": tree_loss.detach(),
                    "recon_loss": (
                        recon_loss.detach()
                        if isinstance(recon_loss, torch.Tensor)
                        else recon_loss
                    ),
                    "div_loss": (
                        div_loss.detach()
                        if isinstance(div_loss, torch.Tensor)
                        else div_loss
                    ),
                    "steps": new_steps.float().sum(),
                    "expand_depth": child_depth_tensor.sum(),
                }

                # Depth-bucketed metrics: [1, mid], [mid+1, max], {mid}, {max}
                mid = self.config.halt_max_steps // 2
                max_d = self.config.halt_max_steps
                buckets = {
                    f"d1_{mid}": (child_depth_tensor >= 1) & (child_depth_tensor <= mid),
                    f"d{mid + 1}_{max_d}": (child_depth_tensor > mid) & (child_depth_tensor <= max_d),
                    f"d{mid}": child_depth_tensor == mid,
                    f"d{max_d}": child_depth_tensor == max_d,
                }
                best_reward = rewards.gather(1, best_idx.view(B, 1)).squeeze(1)  # [B]
                for suffix, mask_b in buckets.items():
                    cnt = mask_b.sum()
                    metrics[f"count_{suffix}"] = cnt
                    metrics[f"accuracy_{suffix}"] = (best_token_acc * mask_b).sum()
                    metrics[f"exact_accuracy_{suffix}"] = (best_exact_match * mask_b).sum()
                    metrics[f"reward_{suffix}"] = (best_reward * mask_b).sum()

        # Best carry from tree
        if best_idx is None:
            best_idx = torch.zeros(B, dtype=torch.long, device=device)

        best_zH = []
        best_zL = []
        for b in range(B):
            best_leaf = self._mcts.get_best_leaf(b)
            best_zH.append(best_leaf.carry.z_H.squeeze(0))
            best_zL.append(best_leaf.carry.z_L.squeeze(0))

        valid_inner_carry = QRMInnerCarry(
            z_H=torch.stack(best_zH),
            z_L=torch.stack(best_zL),
        )

        # Check completion
        all_finish = (new_steps >= self.config.num_iterations).all()
        new_steps = torch.where(all_finish, torch.zeros_like(new_steps), new_steps)
        if all_finish:
            # Tree max_depth: walk tree to find deepest expanded node per batch element
            if metrics is not None:
                max_depths = []
                for b in range(B):
                    stack = [self._mcts.roots[b]]
                    deepest = 0
                    while stack:
                        nd = stack.pop()
                        if nd.depth > deepest:
                            deepest = nd.depth
                        stack.extend(nd.children)
                    max_depths.append(deepest)
                metrics["max_depth"] = torch.tensor(
                    max_depths, dtype=torch.float32, device=device,
                ).sum()
                metrics["count_final"] = torch.tensor(B, device=device)
            # Break parent/child cycles so the tree's CUDA carries (~GB per
            # search) are freed immediately. Without release(), they wait for
            # Python's mark-sweep GC, which isn't triggered by GPU allocations.
            self._mcts.release()
            self._mcts = None

        self._carry = QRMCarry(
            inner=valid_inner_carry, steps=new_steps, current_data=new_current_data,
        )

        return self._make_output(
            loss, logits, z_continuous, z_quantized, indices, weights, priors,
            metrics, all_finish, return_dict,
        )

    def _forward_inference(self, B, device, carry, inner_carry, new_current_data, return_dict):
        """Inference: simple sequential forward (greedy or sampled, no tree)."""

        # Single candidate: greedy or sampled
        do_sampling = self.config.do_sampling
        inner_carry_out, logits, z_continuous, z_quantized, indices, _ = self.model(
            inner_carry, new_current_data,
            return_top_k=False, do_sampling=do_sampling,
        )

        new_steps = carry.steps + 1
        loss = None
        metrics = None
        weights = None

        _, M, N, V = logits.shape  # M=1

        if "labels" in new_current_data and new_current_data["labels"] is not None:
            current_labels = new_current_data["labels"]

            with torch.no_grad():
                rewards, token_acc, exact_match_vals = self._compute_reward(
                    logits, current_labels,
                )

            depth_ratio = new_steps.float().unsqueeze(1) / self.config.halt_max_steps
            depth_penalty = 1.0 - self.config.alpha * depth_ratio
            weights = (0.2 + 0.8 * rewards) * depth_penalty

            tree_loss = qrm_tree_loss(logits, current_labels, weights)

            recon_loss = torch.tensor(0.0, device=logits.device)
            if self.config.lambda_recon > 0:
                z_quantized_mean = z_quantized.mean(dim=1)
                recon_loss = F.mse_loss(z_quantized_mean, z_continuous)

            loss = (
                self.config.lambda_tree * tree_loss
                + self.config.lambda_recon * recon_loss
            )

            with torch.no_grad():
                preds = logits[:, 0].argmax(dim=-1)  # [B, N]
                mask = current_labels != IGNORE_LABEL_ID
                valid_counts = mask.sum(dim=-1).clamp(min=1)
                correct = (preds == current_labels) & mask
                acc = correct.sum(dim=-1).float() / valid_counts
                em = (correct.sum(dim=-1) == valid_counts).float()

                metrics = {
                    "count": torch.tensor(B, device=logits.device),
                    "accuracy": acc.sum(),
                    "exact_accuracy": em.sum(),
                    "mean_reward": rewards.mean(dim=1).sum(),
                    "tree_loss": tree_loss.detach(),
                    "recon_loss": (
                        recon_loss.detach()
                        if isinstance(recon_loss, torch.Tensor)
                        else recon_loss
                    ),
                    "steps": new_steps.float().sum(),
                }

        # M=1, select the only candidate
        valid_inner_carry = QRMInner.select_best_carry(
            inner_carry_out,
            torch.zeros(B, dtype=torch.long, device=device),
        )

        all_finish = (new_steps >= self.config.halt_max_steps).all()
        new_steps = torch.where(all_finish, torch.zeros_like(new_steps), new_steps)

        self._carry = QRMCarry(
            inner=valid_inner_carry, steps=new_steps, current_data=new_current_data,
        )

        return self._make_output(
            loss, logits, z_continuous, z_quantized, indices, weights, None,
            metrics, all_finish, return_dict,
        )

    def _make_output(self, loss, logits, z_continuous, z_quantized, indices,
                     weights, priors, metrics, all_finish, return_dict):
        if not return_dict:
            return (
                loss, logits, z_continuous, z_quantized, indices,
                weights, priors, metrics, all_finish,
            )
        return QRMOutput(
            loss=loss,
            logits=logits,
            z_continuous=z_continuous,
            z_quantized=z_quantized,
            indices=indices,
            weights=weights,
            priors=priors,
            metrics=metrics,
            all_finish=all_finish,
        )
