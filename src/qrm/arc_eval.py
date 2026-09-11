"""ARC Evaluation Script - Run inference on a single checkpoint and save predictions.

Replicates TRM pretrain.py evaluation logic, calling evaluator.update_batch() per batch.

Usage:
    python -m qrm.arc_eval --ckpt_dir results/arc_agi/run_xxx/checkpoint-10000

Output:
    {ckpt_dir}/arc_eval_state.pt - Evaluator state for later aggregation
    {ckpt_dir}/arc_metrics.json - Single checkpoint pass@K metrics
    {ckpt_dir}/arc_results/submission.json - Single checkpoint submission file
"""

import argparse
import json
import logging
import os
from os.path import join

import torch
import yaml
from tqdm import tqdm

from qrm.data.puzzle_dataset import TRMDatasetConfig, TRMIterableDataset
from qrm.evaluators.arc import ARC
from qrm.models.trm.configuration_trm import TRMConfig
from qrm.models.trm.modeling_trm import TRMForPuzzleSolving
from qrm.trainers.trm import TRMTrainer, _clean_param_name, log_weight_comparison

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def passthrough_collator(features):
    """Passthrough collator for IterableDataset that yields pre-batched data."""
    assert (
        len(features) == 1
    ), "IterableDataset with batch_size=1 should yield one batch at a time"
    return features[0]


def run_arc_evaluation(trainer: TRMTrainer, evaluator: ARC, device: torch.device):
    """Run ARC evaluation loop.

    Reuses Trainer's EMA and dataloader creation logic, with custom inference loop
    to support per-batch evaluator.update_batch() calls.

    Args:
        trainer: TRMTrainer instance with loaded checkpoint and EMA.
        evaluator: ARC evaluator instance.
        device: Device to run evaluation on.
    """
    dataloader = trainer.get_eval_dataloader()

    model = trainer.model
    if hasattr(model, "module"):
        model = model.module

    # Apply EMA weights (single eval script, no need to swap back)
    if trainer.use_ema and trainer.ema_helper is not None:
        if trainer.eval_use_compile:
            trainer.ema_helper.swap_weights(model)
            logging.info(
                f"[ARC Eval] Using EMA weights with compiled model (mu={trainer.ema_helper.mu})"
            )
            log_weight_comparison(
                params_a={_clean_param_name(n): p for n, p in model.named_parameters()},
                params_b=trainer.ema_helper.shadow,
                label_a="Eval (EMA)",
                label_b="Train (in shadow)",
                prefix="[ARC Eval]",
            )
        else:
            trainer.ema_helper.ema(model)
            logging.info(f"[ARC Eval] Using EMA weights (mu={trainer.ema_helper.mu})")
            log_weight_comparison(
                params_a={_clean_param_name(n): p for n, p in model.named_parameters()},
                params_b=trainer.ema_helper.shadow,
                label_a="Eval (EMA)",
                label_b="Shadow (EMA)",
                prefix="[ARC Eval]",
            )
    else:
        logging.info("[ARC Eval] Using original model for evaluation (no EMA)")

    model.eval()
    evaluator.begin_eval()
    processed_batches = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="ARC Eval"):
            processed_batches += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)

            while True:
                outputs = model(carry=carry, batch=batch, return_dict=True)
                carry = model._carry
                if outputs.all_finish:
                    break

            preds = {
                "preds": outputs.logits.argmax(-1),
                "q_halt_logits": outputs.q_halt_logits,
            }
            evaluator.update_batch(batch, preds)

    logging.info(f"Processed {processed_batches} batches")


def main():
    parser = argparse.ArgumentParser(
        description="ARC Evaluation for a single checkpoint"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Checkpoint directory path"
    )
    parser.add_argument(
        "--eval_use_compile",
        action="store_true",
        help="Use torch.compile for evaluation (swap_weights instead of ema)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-evaluation even if results exist"
    )
    args = parser.parse_args()

    # Check if already evaluated
    output_path = join(args.ckpt_dir, "arc_eval_state.pt")
    if os.path.exists(output_path) and not args.force:
        logging.info(
            f"Skipping {args.ckpt_dir} (already evaluated, use --force to override)"
        )
        return

    # Load experiment config
    exp_config_path = join(args.ckpt_dir, "../config.yaml")
    if not os.path.exists(exp_config_path):
        exp_config_path = join(args.ckpt_dir, "../../config.yaml")
    if not os.path.exists(exp_config_path):
        raise FileNotFoundError(
            f"Cannot find config.yaml in parent directories of {args.ckpt_dir}"
        )

    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    data_path = exp_config["dataset"]["dataset_paths"][0]
    logging.info(f"Using data path: {data_path}")

    # Verify required data files exist
    identifiers_path = join(data_path, "identifiers.json")
    test_puzzles_path = join(data_path, "test_puzzles.json")
    if not os.path.exists(identifiers_path):
        raise FileNotFoundError(f"identifiers.json not found at {identifiers_path}")
    if not os.path.exists(test_puzzles_path):
        raise FileNotFoundError(f"test_puzzles.json not found at {test_puzzles_path}")

    # Create test dataset
    eval_config = TRMDatasetConfig(
        test_set_mode=True,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        **exp_config["dataset"],
    )
    eval_dataset = TRMIterableDataset(eval_config, split="test")

    # Create model config
    config = TRMConfig(
        batch_size=exp_config["dataset"]["global_batch_size"],
        seq_len=eval_dataset.metadata["seq_len"],
        vocab_size=eval_dataset.metadata["vocab_size"],
        num_puzzle_identifiers=eval_dataset.metadata["num_puzzle_identifiers"],
        **exp_config["model"],
    )

    # Initialize model
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with torch.device(device):
        model = TRMForPuzzleSolving(config)

    # Load training args for Trainer initialization
    training_args = torch.load(f"{args.ckpt_dir}/training_args.bin", weights_only=False)
    training_args.report_to = "none"
    training_args.torch_compile = True
    training_args.torch_compile_backend = "inductor"

    # Initialize Trainer (reuses model loading, EMA, and dataloader creation)
    trainer = TRMTrainer(
        model=model,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=passthrough_collator,
        puzzle_emb_lr=exp_config["train"]["puzzle_emb_lr"],
        puzzle_emb_weight_decay=exp_config["train"]["puzzle_emb_weight_decay"],
        lr_min_ratio=exp_config["train"]["lr_min_ratio"],
        ema=exp_config["train"]["ema"],
        ema_rate=exp_config["train"]["ema_rate"],
        eval_use_compile=args.eval_use_compile,
    )
    trainer._load_from_checkpoint(args.ckpt_dir)

    # Create ARC evaluator (single checkpoint mode, no accumulation)
    evaluator = ARC(
        data_path=data_path,
        blank_identifier_id=eval_dataset.metadata["blank_identifier_id"],
        aggregated_voting=False,
    )

    logging.info(f"Running ARC evaluation on {args.ckpt_dir}")
    run_arc_evaluation(trainer, evaluator, device)

    # Save evaluator state for later aggregation
    torch.save(evaluator.state_dict(), output_path)
    logging.info(f"Saved evaluator state to {output_path}")

    # Compute and save single checkpoint results
    results_dir = join(args.ckpt_dir, "arc_results")
    results = evaluator.result(save_path=results_dir)
    logging.info(f"Single checkpoint results: {results}")

    with open(join(args.ckpt_dir, "arc_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved metrics to {join(args.ckpt_dir, 'arc_metrics.json')}")


if __name__ == "__main__":
    main()
