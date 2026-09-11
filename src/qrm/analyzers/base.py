"""Base analyzer for inference analysis."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import yaml

from qrm.data.puzzle_dataset import TRMDatasetConfig, TRMIterableDataset
from qrm.models.trm.configuration_trm import TRMConfig
from qrm.models.trm.modeling_trm import TRMForPuzzleSolving, TRMInnerCarry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Base class for inference analysis.

    Handles checkpoint loading, dataset iteration, and per-round JSON I/O.
    Subclasses implement analyze_round() to define what to record.
    """

    def __init__(
        self,
        ckpt_dir: str,
        output_dir: str,
        max_samples: int = None,
        batch_size: int = 768,
        fsq_sampling_inference: bool = None,
        save_preds: bool = False,
    ):
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.fsq_sampling_inference = fsq_sampling_inference
        self.save_preds = save_preds
        self.model = None
        self.config = None
        self.dataset = None
        self.exp_config = None

    def load(self):
        """Load checkpoint, config, dataset, and model weights."""
        exp_config_path = os.path.join(self.ckpt_dir, "../config.yaml")
        with open(exp_config_path, "r") as f:
            self.exp_config = yaml.safe_load(f)

        # Override global_batch_size with analysis batch_size
        dataset_cfg = {**self.exp_config["dataset"]}
        dataset_cfg["global_batch_size"] = self.batch_size

        eval_config = TRMDatasetConfig(
            test_set_mode=True,
            rank=0,
            num_replicas=1,
            epochs_per_iter=1,
            **dataset_cfg,
        )
        self.dataset = TRMIterableDataset(eval_config, split="test")

        model_cfg = self.exp_config["model"]
        self.config = TRMConfig(
            batch_size=self.batch_size,
            seq_len=self.dataset.metadata["seq_len"],
            vocab_size=self.dataset.metadata["vocab_size"],
            num_puzzle_identifiers=self.dataset.metadata["num_puzzle_identifiers"],
            **model_cfg,
        )

        # Apply fsq_sampling_inference override if provided
        if self.fsq_sampling_inference is not None and self.config.use_fsq:
            logger.info(
                f"Overriding fsq_sampling_inference: "
                f"{self.config.fsq_sampling_inference} -> {self.fsq_sampling_inference}"
            )
            self.config.fsq_sampling_inference = self.fsq_sampling_inference

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
            logger.info("No CUDA available, using CPU")
        with torch.device(device):
            self.model = TRMForPuzzleSolving(self.config)

        # Load weights
        from safetensors.torch import load_file

        state_dict = load_file(os.path.join(self.ckpt_dir, "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=False)

        # Load EMA weights if available
        # ema_state.pt has structure {"shadow": {param_name: tensor, ...}, "mu": float}
        ema_path = os.path.join(self.ckpt_dir, "ema_state.pt")
        if os.path.exists(ema_path):
            logger.info("Loading EMA weights")
            ema_state = torch.load(ema_path, map_location=device, weights_only=True)
            shadow = ema_state["shadow"]
            model_state = self.model.state_dict()
            replaced = 0
            for k, v in shadow.items():
                if k in model_state:
                    model_state[k] = v
                    replaced += 1
            self.model.load_state_dict(model_state)
            logger.info(f"Replaced {replaced}/{len(shadow)} params with EMA weights")

        self.model.eval()
        logger.info(f"Loaded model from {self.ckpt_dir} (batch_size={self.batch_size})")

    @abstractmethod
    def analyze_round(
        self,
        round_idx: int,
        model,
        carry: TRMInnerCarry,
        batch: dict,
        labels: torch.Tensor,
    ) -> Tuple[TRMInnerCarry, List[dict]]:
        """Analyze a single round for a batch of samples.

        Args:
            round_idx: Current round index (0-based)
            model: The inner model (TRMInner)
            carry: Current inner carry state [B, ...]
            batch: Input batch [B, ...]
            labels: Ground truth labels [B, seq_len]

        Returns:
            new_carry: Updated carry for next round
            sample_results: List of B dicts, one per sample in the batch
        """
        ...

    def run(self):
        """Run full analysis."""
        self.load()
        os.makedirs(self.output_dir, exist_ok=True)

        # Build config summary for JSON output
        config_summary = {
            "H_cycles": self.config.H_cycles,
            "L_cycles": self.config.L_cycles,
            "halt_max_steps": self.config.halt_max_steps,
            "use_fsq": self.config.use_fsq,
        }
        if self.config.use_fsq:
            config_summary.update({
                "fsq_levels": self.config.fsq_levels,
                "fsq_top_k": self.config.fsq_top_k,
                "fsq_sampling_inference": self.config.fsq_sampling_inference,
            })

        halt_max_steps = self.config.halt_max_steps
        round_data = {
            r: {"round": r + 1, "config": config_summary, "results": None, "samples": []}
            for r in range(halt_max_steps)
        }
        metadata = {"samples": []}

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            self.dataset, batch_size=None, collate_fn=lambda x: x
        )

        with torch.inference_mode():
            inner_model = self.model.model  # TRMInner
            device = next(self.model.parameters()).device

            from tqdm import tqdm

            sample_offset = 0
            pbar = tqdm(desc="Analyzing", total=self.max_samples)

            for full_batch in dataloader:
                full_batch = {k: v.to(device) for k, v in full_batch.items()}
                B = full_batch["input_ids"].shape[0]

                # Trim batch if we'd exceed max_samples
                if self.max_samples and sample_offset + B > self.max_samples:
                    remaining = self.max_samples - sample_offset
                    full_batch = {k: v[:remaining] for k, v in full_batch.items()}
                    B = remaining

                # Save metadata
                for i in range(B):
                    metadata["samples"].append({
                        "sample_id": sample_offset + i,
                        "input_ids": full_batch["input_ids"][i].tolist(),
                        "labels": full_batch["labels"][i].tolist(),
                    })

                # Initialize carry for full batch
                carry = self.model.initial_carry(full_batch)
                inner_carry = inner_model.reset_carry(carry.halted, carry.inner)

                # Run all rounds
                for round_idx in range(halt_max_steps):
                    new_inner_carry, sample_results = self.analyze_round(
                        round_idx, inner_model, inner_carry,
                        full_batch, full_batch["labels"]
                    )
                    # Tag each result with sample_id
                    for i, result in enumerate(sample_results):
                        result["sample_id"] = sample_offset + i
                    round_data[round_idx]["samples"].extend(sample_results)
                    inner_carry = TRMInnerCarry(
                        z_H=new_inner_carry.z_H.detach(),
                        z_L=new_inner_carry.z_L.detach(),
                    )

                sample_offset += B
                pbar.update(B)

                if self.max_samples and sample_offset >= self.max_samples:
                    break

            pbar.close()

        # Compute results summary and save
        logger.info("Computing results summaries...")
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        for round_idx, data in round_data.items():
            # Filter out padded samples for summary statistics
            real_samples = [s for s in data["samples"] if not s.get("is_pad", False)]
            num_real = len(real_samples)
            if num_real == 0:
                continue

            num_h_cycles = len(real_samples[0]["h_cycles"])
            h_cycle_summary = []
            for h in range(num_h_cycles):
                token_accs = [s["h_cycles"][h]["token_acc"] for s in real_samples]
                exact_matches = [s["h_cycles"][h]["exact_match"] for s in real_samples]
                h_cycle_summary.append({
                    "h_cycle": h,
                    "mean_token_acc": round(sum(token_accs) / num_real, 4),
                    "exact_accuracy": round(sum(exact_matches) / num_real, 4),
                })

            data["results"] = {
                "num_samples": num_real,
                "num_samples_with_pad": len(data["samples"]),
                "h_cycles": h_cycle_summary,
            }

            path = os.path.join(self.output_dir, f"round_{round_idx + 1:02d}.json")
            with open(path, "w") as f:
                json.dump(data, f)
            logger.info(
                f"  Round {round_idx + 1}: "
                + ", ".join(
                    f"h{h['h_cycle']}={h['exact_accuracy']:.2%}"
                    for h in h_cycle_summary
                )
            )

        logger.info(f"Analysis saved to {self.output_dir}")
