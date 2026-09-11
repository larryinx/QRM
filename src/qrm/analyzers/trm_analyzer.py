"""TRM-specific inference analyzer.

Records z_H intermediate states at each H_cycle and FSQ behavior.
"""

from typing import List, Tuple

import torch

from qrm.losses.trm import IGNORE_LABEL_ID
from qrm.models.trm.modeling_trm import TRMInnerCarry

from .base import BaseAnalyzer


class TRMAnalyzer(BaseAnalyzer):
    """TRM analysis: z_H intermediate states + FSQ behavior."""

    def analyze_round(
        self, round_idx, model, carry, batch, labels
    ) -> Tuple[TRMInnerCarry, List[dict]]:
        B = batch["input_ids"].shape[0]

        input_embeddings = model._input_embeddings(
            batch["input_ids"], batch.get("puzzle_identifiers")
        )
        cos_sin = model.rotary_emb() if hasattr(model, "rotary_emb") else None

        z_H = carry.z_H
        z_L = carry.z_L

        # --- Phase 1: All GPU work, store raw tensors ---
        h_cycle_preds = []  # List of (h_cycle_idx, preds_tensor)

        # H_cycles - 1 (intermediate cycles)
        for _h in range(model.config.H_cycles - 1):
            for _l in range(model.config.L_cycles):
                z_L = model.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
            z_H = model.L_level(z_H, z_L, cos_sin=cos_sin)

            logits = model.lm_head(z_H)[:, model.puzzle_emb_len :]
            h_cycle_preds.append((_h, logits.argmax(-1)))  # [B, seq_len]

        # Last H_cycle: L_cycles
        for _l in range(model.config.L_cycles):
            z_L = model.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

        # FSQ (if enabled)
        fsq_diagnostics = None
        if model.config.use_fsq:
            do_sampling = (
                model.config.fsq_sampling_training
                if model.training
                else model.config.fsq_sampling_inference
            )
            z_L_quantized, _, fsq_diagnostics = model.fsq(
                z_L,
                return_top_k=False,
                do_sampling=do_sampling,
                return_diagnostics=True,
            )
            z_L_quantized = z_L_quantized.reshape(z_L.shape)

            if hasattr(model, "fsq_residual_weight"):
                if model.config.fsq_residual_mode == "fixed":
                    alpha = model.fsq_residual_weight
                else:
                    alpha = torch.sigmoid(model.fsq_residual_weight)
                z_L_mixed = alpha * z_L_quantized + (1 - alpha) * z_L
            else:
                z_L_mixed = z_L_quantized

            # Match model's forward: carry gets post-FSQ z_L (preserves bottleneck)
            z_L = z_L_mixed
            z_H = model.L_level(z_H, z_L, cos_sin=cos_sin)
        else:
            z_H = model.L_level(z_H, z_L, cos_sin=cos_sin)

        # Final H_cycle decode
        logits = model.lm_head(z_H)[:, model.puzzle_emb_len :]
        h_cycle_preds.append((model.config.H_cycles - 1, logits.argmax(-1)))

        # Q-halt logits [B]
        q_halt_logits = model.q_head(z_H[:, 0]).to(torch.float32)[:, 0]  # [B]

        new_carry = TRMInnerCarry(z_H=z_H, z_L=z_L)

        # --- Phase 2: Bulk GPU->CPU, then Python assembly ---
        # Compute all metrics on GPU first, then transfer once
        mask = labels != IGNORE_LABEL_ID  # [B, seq_len]
        valid_count = mask.sum(-1)  # [B]

        # Stack all h_cycle preds into one tensor for bulk transfer
        all_preds = torch.stack([p for _, p in h_cycle_preds])  # [num_h, B, seq_len]
        all_correct = mask.unsqueeze(0) & (all_preds == labels.unsqueeze(0))  # [num_h, B, seq_len]
        all_correct_count = all_correct.sum(-1)  # [num_h, B]
        all_token_accs = all_correct_count.float() / valid_count.unsqueeze(0).clamp_min(1)  # [num_h, B]
        # Exclude padded samples (valid_count == 0): 0 == 0 would be a false positive
        valid_sample = valid_count > 0  # [B]
        all_exact = (all_correct_count == valid_count.unsqueeze(0)) & valid_sample.unsqueeze(0)  # [num_h, B]

        # Round token_accs on GPU: multiply, round, divide
        all_token_accs = (all_token_accs * 10000).round() / 10000

        # Bulk CPU transfer (skip preds unless requested — it's the largest tensor)
        all_preds_cpu = all_preds.cpu().tolist() if self.save_preds else None
        all_token_accs_cpu = all_token_accs.cpu().tolist() # [num_h][B]
        all_exact_cpu = all_exact.cpu().tolist()           # [num_h][B]
        q_halt_cpu = ((q_halt_logits * 10000).round() / 10000).tolist()  # [B]
        valid_sample_cpu = valid_sample.tolist()  # [B]

        h_cycle_indices = [idx for idx, _ in h_cycle_preds]
        num_h = len(h_cycle_indices)

        # FSQ: transpose and round on GPU before transfer
        fsq_per_sample = None
        if fsq_diagnostics is not None:
            fsq_per_sample = self._extract_fsq_data_batch(fsq_diagnostics, model)

        # Assemble per-sample results
        sample_results = []
        for i in range(B):
            h_cycles = []
            for h in range(num_h):
                h_dict = {
                    "h_cycle": h_cycle_indices[h],
                    "token_acc": all_token_accs_cpu[h][i],
                    "exact_match": all_exact_cpu[h][i],
                }
                if all_preds_cpu is not None:
                    h_dict["preds"] = all_preds_cpu[h][i]
                h_cycles.append(h_dict)
            result = {
                "q_halt_logit": q_halt_cpu[i],
                "is_pad": not valid_sample_cpu[i],
                "h_cycles": h_cycles,
            }
            if fsq_per_sample is not None:
                result["fsq"] = fsq_per_sample[i]
            sample_results.append(result)

        return new_carry, sample_results

    def _extract_fsq_data_batch(self, diagnostics: dict, model) -> List[list]:
        """Extract FSQ diagnostics per sample in the batch."""
        puzzle_emb_len = model.puzzle_emb_len

        # [B, seq_len, top_k]
        topk_indices = diagnostics["topk_indices"][:, puzzle_emb_len:]
        topk_log_probs = diagnostics["topk_log_probs"][:, puzzle_emb_len:]
        selected = diagnostics["selected_indices"][:, puzzle_emb_len:, 0]  # [B, seq_len]

        # is_chosen: [B, top_k]
        is_chosen_all = (topk_indices == selected.unsqueeze(-1)).all(dim=1)

        # Transpose on GPU: [B, seq_len, top_k] -> [B, top_k, seq_len]
        topk_indices_t = topk_indices.permute(0, 2, 1).contiguous()
        topk_log_probs_t = topk_log_probs.permute(0, 2, 1).contiguous()

        # Round log_probs on GPU
        topk_log_probs_t = (topk_log_probs_t.float() * 10000).round() / 10000

        # Bulk CPU transfer (already in [B, top_k, seq_len] layout)
        indices_cpu = topk_indices_t.cpu().tolist()     # [B][top_k][seq_len]
        log_probs_cpu = topk_log_probs_t.cpu().tolist() # [B][top_k][seq_len]
        is_chosen_cpu = is_chosen_all.tolist()           # [B][top_k]

        B = topk_indices.shape[0]
        top_k = topk_indices.shape[2]

        batch_fsq = []
        for i in range(B):
            fsq_list = []
            for k in range(top_k):
                fsq_list.append({
                    "rank": k,
                    "is_chosen": is_chosen_cpu[i][k],
                    "flat_indices": indices_cpu[i][k],
                    "log_probs": log_probs_cpu[i][k],
                })
            batch_fsq.append(fsq_list)

        return batch_fsq
