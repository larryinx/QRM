"""FSQ entropy collapse diagnostic analyzer.

Computes codebook utilization, per-position entropy, top-k gap,
per-dimension level usage, and quantization error from existing checkpoints.
All statistics are accumulated on GPU and saved as a single summary JSON.
"""

import json
import logging
import math
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F

from qrm.losses.trm import IGNORE_LABEL_ID
from qrm.models.trm.modeling_trm import TRMInnerCarry

from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class FSQEntropyAnalyzer(BaseAnalyzer):
    """FSQ entropy collapse diagnostic analyzer.

    Computes 5 diagnostic metrics (D1-D5) aggregated across batches on GPU.
    Only analyzes at selected rounds (default: 1, 8, 16) but runs all rounds
    to maintain correct carry state.
    """

    def __init__(
        self,
        ckpt_dir: str,
        output_dir: str,
        max_samples: int = None,
        batch_size: int = 768,
        fsq_sampling_inference: bool = None,
        save_preds: bool = False,
        analyze_rounds: tuple = (0, 7, 15),  # 0-indexed
    ):
        super().__init__(
            ckpt_dir, output_dir, max_samples, batch_size,
            fsq_sampling_inference, save_preds,
        )
        self.analyze_rounds = set(analyze_rounds)

    def analyze_round(self, *args, **kwargs):
        raise NotImplementedError("FSQEntropyAnalyzer uses run() directly")

    def _run_reasoning_and_fsq(self, model, z_H, z_L, input_embeddings, cos_sin, collect_diagnostics):
        """Run H/L cycles and FSQ, optionally collecting diagnostics.

        Returns:
            (z_H, z_L, diagnostics_or_None, z_L_pre_fsq_or_None)
        """
        # H_cycles - 1 (intermediate, no grad already via inference_mode)
        for _h in range(model.config.H_cycles - 1):
            for _l in range(model.config.L_cycles):
                z_L = model.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
            z_H = model.L_level(z_H, z_L, cos_sin=cos_sin)

        # Last H_cycle: L_cycles
        for _l in range(model.config.L_cycles):
            z_L = model.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

        # FSQ
        fsq_diag = None
        z_L_pre = None
        if model.config.use_fsq:
            if collect_diagnostics:
                z_L_pre = z_L  # save reference before FSQ

            do_sampling = (
                model.config.fsq_sampling_training
                if model.training
                else model.config.fsq_sampling_inference
            )
            if collect_diagnostics:
                z_L_quantized, _, fsq_diag = model.fsq(
                    z_L, return_top_k=False, do_sampling=do_sampling,
                    return_diagnostics=True,
                )
            else:
                z_L_quantized, _ = model.fsq(
                    z_L, return_top_k=False, do_sampling=do_sampling,
                )
            z_L_quantized = z_L_quantized.reshape(z_L.shape)

            # Residual mixing
            if hasattr(model, "fsq_residual_weight"):
                if model.config.fsq_residual_mode == "fixed":
                    alpha = model.fsq_residual_weight
                elif model.config.fsq_residual_mode == "learned_scalar":
                    alpha = torch.sigmoid(model.fsq_residual_weight)
                else:
                    alpha = model.fsq_residual_weight
                z_L = alpha * z_L_quantized + (1 - alpha) * z_L
            elif hasattr(model, "fsq_attention"):
                # attention residual mode — not expected here but handle gracefully
                z_L = z_L_quantized
            else:
                z_L = z_L_quantized

            z_H = model.L_level(z_H, z_L, cos_sin=cos_sin)
        else:
            z_H = model.L_level(z_H, z_L, cos_sin=cos_sin)

        return z_H, z_L, fsq_diag, z_L_pre

    def run(self):
        """Run FSQ entropy diagnostic analysis."""
        self.load()
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.config.use_fsq:
            logger.error("FSQ is not enabled in this checkpoint. Nothing to analyze.")
            return

        inner_model = self.model.model  # TRMInner
        device = next(self.model.parameters()).device
        halt_max_steps = self.config.halt_max_steps

        # FSQ parameters
        fsq = inner_model.fsq
        codebook_size = fsq.codebook_size
        codebook_dim = fsq.codebook_dim
        levels = fsq.levels
        puzzle_emb_len = inner_model.puzzle_emb_len

        # Clamp analyze_rounds to valid range
        self.analyze_rounds = {r for r in self.analyze_rounds if r < halt_max_steps}

        # Get residual alpha
        residual_info = {"mode": "none", "alpha": None}
        if hasattr(inner_model, "fsq_residual_weight"):
            mode = self.config.fsq_residual_mode
            if mode == "fixed":
                alpha_val = inner_model.fsq_residual_weight.item()
            elif mode == "learned_scalar":
                alpha_val = torch.sigmoid(inner_model.fsq_residual_weight).item()
            else:
                alpha_val = inner_model.fsq_residual_weight.item()
            residual_info = {"mode": mode, "alpha": round(alpha_val, 6)}
        logger.info(f"FSQ residual: {residual_info}")

        # Initialize per-round accumulators
        accumulators = {}
        for r in self.analyze_rounds:
            accumulators[r] = {
                # D1: codebook utilization
                "code_histogram": torch.zeros(codebook_size, device=device, dtype=torch.long),
                # D2: per-position code counts — will be initialized when we know seq_len
                "position_code_counts": None,
                "answer_mask_sum": None,  # how many answer positions per position
                # D3: top-k log probs
                "topk_lp_sum": torch.zeros(fsq.top_k, device=device, dtype=torch.float64),
                "topk_lp_count": 0,
                # D4: per-dim level counts
                "dim_level_counts": [
                    torch.zeros(levels[d], device=device, dtype=torch.long)
                    for d in range(codebook_dim)
                ],
                # D5: quantization error
                "mse_sum": 0.0,
                "cosine_sum": 0.0,
                "quant_count": 0,
                # sample count
                "num_positions": 0,
            }

        # Config summary
        config_summary = {
            "H_cycles": self.config.H_cycles,
            "L_cycles": self.config.L_cycles,
            "halt_max_steps": halt_max_steps,
            "fsq_levels": list(levels),
            "fsq_top_k": fsq.top_k,
            "fsq_temperature": fsq.temperature,
            "fsq_residual": residual_info,
            "codebook_size": codebook_size,
            "codebook_dim": codebook_dim,
            "analyze_rounds": sorted(self.analyze_rounds),
        }

        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.dataset, batch_size=None, collate_fn=lambda x: x)

        from tqdm import tqdm
        sample_offset = 0
        pbar = tqdm(desc="FSQ Entropy Analysis", total=self.max_samples)

        with torch.inference_mode():
            for full_batch in dataloader:
                full_batch = {k: v.to(device) for k, v in full_batch.items()}
                B = full_batch["input_ids"].shape[0]

                if self.max_samples and sample_offset + B > self.max_samples:
                    remaining = self.max_samples - sample_offset
                    full_batch = {k: v[:remaining] for k, v in full_batch.items()}
                    B = remaining

                labels = full_batch["labels"]  # [B, seq_len]
                seq_len = labels.shape[1]
                answer_mask = (labels != IGNORE_LABEL_ID)  # [B, seq_len]

                # Initialize position_code_counts if needed
                for r in self.analyze_rounds:
                    acc = accumulators[r]
                    if acc["position_code_counts"] is None:
                        acc["position_code_counts"] = torch.zeros(
                            seq_len, codebook_size, device=device, dtype=torch.long
                        )
                        acc["answer_mask_sum"] = torch.zeros(
                            seq_len, device=device, dtype=torch.long
                        )

                # Initialize carry
                carry = self.model.initial_carry(full_batch)
                inner_carry = inner_model.reset_carry(carry.halted, carry.inner)

                input_embeddings = inner_model._input_embeddings(
                    full_batch["input_ids"], full_batch.get("puzzle_identifiers")
                )
                cos_sin = inner_model.rotary_emb() if hasattr(inner_model, "rotary_emb") else None

                for round_idx in range(halt_max_steps):
                    collect = round_idx in self.analyze_rounds

                    z_H, z_L, fsq_diag, z_L_pre = self._run_reasoning_and_fsq(
                        inner_model, inner_carry.z_H, inner_carry.z_L,
                        input_embeddings, cos_sin, collect_diagnostics=collect,
                    )

                    if collect and fsq_diag is not None:
                        acc = accumulators[round_idx]

                        # Strip puzzle_emb_len prefix
                        sel_indices = fsq_diag["selected_indices"][:, puzzle_emb_len:, 0]  # [B, seq_len]
                        bounded_z = fsq_diag["bounded_z"][:, puzzle_emb_len:]  # [B, seq_len, codebook_dim]
                        topk_lp = fsq_diag["topk_log_probs"][:, puzzle_emb_len:]  # [B, seq_len, top_k]

                        # D1: codebook histogram
                        acc["code_histogram"].scatter_add_(
                            0, sel_indices.reshape(-1),
                            torch.ones(sel_indices.numel(), device=device, dtype=torch.long),
                        )

                        # D2: per-position code counts
                        for pos in range(seq_len):
                            acc["position_code_counts"][pos].scatter_add_(
                                0, sel_indices[:, pos],
                                torch.ones(B, device=device, dtype=torch.long),
                            )
                        acc["answer_mask_sum"] += answer_mask.sum(dim=0)  # [seq_len]

                        # D3: top-k log probs (mean over batch and positions)
                        acc["topk_lp_sum"] += topk_lp.to(torch.float64).mean(dim=(0, 1))  # [top_k]
                        acc["topk_lp_count"] += 1

                        # D4: per-dimension level histogram from bounded_z
                        levels_tensor = torch.tensor(levels, device=device)
                        half_width = (levels_tensor // 2).float()
                        level_indices = (bounded_z * half_width + half_width).round().long()
                        # Clamp to valid range
                        for d in range(codebook_dim):
                            li = level_indices[:, :, d].clamp(0, levels[d] - 1).reshape(-1)
                            acc["dim_level_counts"][d].scatter_add_(
                                0, li,
                                torch.ones(li.numel(), device=device, dtype=torch.long),
                            )

                        # D5: quantization error in codebook-dim space
                        codes_selected = fsq.implicit_codebook[sel_indices.reshape(-1)]  # [B*seq_len, codebook_dim]
                        bounded_flat = bounded_z.reshape(-1, codebook_dim)  # [B*seq_len, codebook_dim]
                        mse = F.mse_loss(bounded_flat, codes_selected)
                        cos = F.cosine_similarity(bounded_flat, codes_selected, dim=-1).mean()
                        acc["mse_sum"] += mse.item()
                        acc["cosine_sum"] += cos.item()
                        acc["quant_count"] += 1
                        acc["num_positions"] += B * seq_len

                    inner_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

                sample_offset += B
                pbar.update(B)

                if self.max_samples and sample_offset >= self.max_samples:
                    break

        pbar.close()

        # Compute final summary
        logger.info("Computing summary statistics...")
        summary = {
            "config": config_summary,
            "num_samples": sample_offset,
            "per_round": {},
        }

        for r in sorted(self.analyze_rounds):
            acc = accumulators[r]
            if acc["quant_count"] == 0:
                continue

            hist = acc["code_histogram"]
            total_selections = hist.sum().item()

            # D1: Codebook utilization
            active_codes = (hist > 0).sum().item()
            utilization_rate = active_codes / codebook_size
            sorted_hist, _ = hist.sort(descending=True)
            cumsum = sorted_hist.cumsum(0).float() / max(total_selections, 1)
            top10_cover = cumsum[min(9, codebook_size - 1)].item()
            top50_cover = cumsum[min(49, codebook_size - 1)].item()

            # D2: Per-position entropy
            pos_counts = acc["position_code_counts"]  # [seq_len, codebook_size]
            pos_total = pos_counts.sum(dim=-1, keepdim=True).clamp(min=1).float()
            pos_probs = pos_counts.float() / pos_total
            # Entropy: -sum(p * log2(p))
            log_probs = torch.where(pos_probs > 0, pos_probs.log2(), torch.zeros_like(pos_probs))
            pos_entropy = -(pos_probs * log_probs).sum(dim=-1)  # [seq_len]

            answer_mask_sum = acc["answer_mask_sum"]  # [seq_len]
            is_answer_pos = answer_mask_sum > 0
            is_prompt_pos = ~is_answer_pos

            mean_entropy = pos_entropy.mean().item()
            answer_entropy = pos_entropy[is_answer_pos].mean().item() if is_answer_pos.any() else 0.0
            prompt_entropy = pos_entropy[is_prompt_pos].mean().item() if is_prompt_pos.any() else 0.0
            max_possible_entropy = math.log2(codebook_size)

            # D3: Top-k gap
            mean_topk_lp = acc["topk_lp_sum"] / acc["topk_lp_count"]
            mean_topk_prob = mean_topk_lp.exp()
            top1_prob = mean_topk_prob[0].item()
            gap_1_2 = (mean_topk_lp[0] - mean_topk_lp[1]).item() if fsq.top_k > 1 else 0.0
            gap_1_k = (mean_topk_lp[0] - mean_topk_lp[-1]).item() if fsq.top_k > 1 else 0.0

            # D4: Per-dimension level entropy
            d4_dims = []
            effective_codebook_size = 1
            for d in range(codebook_dim):
                counts = acc["dim_level_counts"][d]
                total = counts.sum().clamp(min=1).float()
                p = counts.float() / total
                log_p = torch.where(p > 0, p.log2(), torch.zeros_like(p))
                dim_entropy = -(p * log_p).sum().item()
                max_dim_entropy = math.log2(levels[d])
                active_levels = (counts > 0).sum().item()
                effective_codebook_size *= active_levels
                d4_dims.append({
                    "dim": d,
                    "levels": levels[d],
                    "entropy": round(dim_entropy, 4),
                    "max_entropy": round(max_dim_entropy, 4),
                    "active_levels": active_levels,
                    "histogram": counts.tolist(),
                })

            # D5: Quantization error
            mean_mse = acc["mse_sum"] / acc["quant_count"]
            mean_cos = acc["cosine_sum"] / acc["quant_count"]

            # Verdicts
            def utilization_verdict(rate):
                if rate < 0.10:
                    return "SEVERE_COLLAPSE"
                if rate < 0.30:
                    return "MODERATE_COLLAPSE"
                if rate < 0.50:
                    return "MILD_UNDERUTILIZATION"
                return "HEALTHY"

            def entropy_verdict(ent, max_ent):
                ratio = ent / max_ent if max_ent > 0 else 0
                if ratio < 0.30:
                    return "SEVERE_COLLAPSE"
                if ratio < 0.50:
                    return "MODERATE_COLLAPSE"
                return "HEALTHY"

            round_summary = {
                "round": r + 1,
                "d1_codebook_utilization": {
                    "active_codes": active_codes,
                    "codebook_size": codebook_size,
                    "utilization_rate": round(utilization_rate, 4),
                    "top10_codes_cover_pct": round(top10_cover, 4),
                    "top50_codes_cover_pct": round(top50_cover, 4),
                    "verdict": utilization_verdict(utilization_rate),
                },
                "d2_position_entropy": {
                    "mean_entropy": round(mean_entropy, 4),
                    "answer_positions_mean": round(answer_entropy, 4),
                    "prompt_positions_mean": round(prompt_entropy, 4),
                    "min_entropy": round(pos_entropy.min().item(), 4),
                    "max_entropy": round(pos_entropy.max().item(), 4),
                    "max_possible_entropy": round(max_possible_entropy, 4),
                    "num_answer_positions": int(is_answer_pos.sum().item()),
                    "num_prompt_positions": int(is_prompt_pos.sum().item()),
                    "verdict": entropy_verdict(mean_entropy, max_possible_entropy),
                },
                "d3_topk_gap": {
                    "mean_top1_prob": round(top1_prob, 6),
                    "mean_gap_1_2": round(gap_1_2, 4),
                    "mean_gap_1_k": round(gap_1_k, 4),
                    "mean_topk_probs": [round(p, 6) for p in mean_topk_prob.tolist()],
                },
                "d4_per_dim_levels": {
                    "dims": d4_dims,
                    "effective_codebook_size": effective_codebook_size,
                },
                "d5_quantization_error": {
                    "mean_mse": round(mean_mse, 6),
                    "mean_cosine_similarity": round(mean_cos, 6),
                },
            }

            summary["per_round"][f"round_{r + 1:02d}"] = round_summary

            logger.info(
                f"  Round {r + 1}: "
                f"active={active_codes}/{codebook_size} ({utilization_rate:.1%}), "
                f"entropy={mean_entropy:.2f}/{max_possible_entropy:.2f}, "
                f"top1_prob={top1_prob:.4f}, "
                f"cos_sim={mean_cos:.4f}"
            )

        # Save summary
        out_path = os.path.join(self.output_dir, "fsq_entropy_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {out_path}")
