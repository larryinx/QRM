"""ARC Aggregation Script - Aggregate predictions from multiple checkpoints.

Implements TRM's Aggregated Voting mechanism: accumulate predictions across checkpoints
and vote for final answers.

Usage:
    python -m qrm.arc_aggregate --exp_dir results/arc_agi/run_xxx

Output:
    {exp_dir}/arc_aggregate_ckpt{N}/arc_metrics.json - Aggregated pass@K metrics
    {exp_dir}/arc_aggregate_ckpt{N}/submission.json - Kaggle format submission file
    (N is the actual number of aggregated checkpoints)
"""

import argparse
import json
import logging
import os
from glob import glob
from os.path import join

import torch
import yaml

from qrm.data.puzzle_dataset import TRMDatasetConfig, TRMIterableDataset
from qrm.evaluators.arc import ARC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _build_wandb_id(run_name: str) -> str:
    """Mirror the id formula used by qrm.train so we resume the same run."""
    existing = os.environ.get("WANDB_RUN_ID")
    if existing:
        return existing
    project = os.environ.get("WANDB_PROJECT", "")
    raw = f"{project}__{run_name}" if project else run_name
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:128]


def _log_aggregate_to_wandb(run_name, results, num_ckpts, output_dir):
    """Resume the training run and log aggregated metrics on an ensemble-size axis."""
    try:
        import wandb
    except ImportError:
        logging.warning("[ARC Aggregate] wandb not installed, skipping upload")
        return

    project = os.environ.get("WANDB_PROJECT")
    if not project:
        logging.warning(
            "[ARC Aggregate] WANDB_PROJECT not set, skipping wandb upload"
        )
        return

    run_id = _build_wandb_id(run_name)
    logging.info(
        f"[ARC Aggregate] Resuming wandb run {project}/{run_id} (num_ckpts={num_ckpts})"
    )

    try:
        run = wandb.init(project=project, id=run_id, resume="must", reinit=True)
    except Exception as e:
        logging.warning(
            f"[ARC Aggregate] Resume failed for {project}/{run_id}: {e}. "
            f"Creating a new run with the same id."
        )
        run = wandb.init(project=project, id=run_id, resume="allow", reinit=True)

    wandb.define_metric("arc_agg/num_ckpts")
    wandb.define_metric("arc_agg/*", step_metric="arc_agg/num_ckpts")

    log_dict = {}
    for k, v in results.items():
        suffix = k.split("/", 1)[1] if "/" in k else k
        log_dict[f"arc_agg/{suffix}"] = v
    log_dict["arc_agg/num_ckpts"] = num_ckpts

    wandb.log(log_dict)

    # Mirror to summary so the aggregated pass@K stays visible without digging into history.
    for k, v in results.items():
        suffix = k.split("/", 1)[1] if "/" in k else k
        run.summary[f"arc_agg_latest/{suffix}"] = v
    run.summary["arc_agg_latest/num_ckpts"] = num_ckpts
    run.summary["arc_agg_latest/output_dir"] = os.path.abspath(output_dir)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ARC predictions from multiple checkpoints"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory containing checkpoint-* subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {exp_dir}/arc_aggregate)",
    )
    parser.add_argument(
        "--pass_ks",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 100, 1000],
        help="Pass@K values to compute (default: 1 2 5 10 100 1000)",
    )
    parser.add_argument(
        "--submission_k",
        type=int,
        default=2,
        help="Top-K predictions for submission (default: 2)",
    )
    parser.add_argument(
        "--ckpt_pattern",
        type=str,
        default="checkpoint-*",
        help="Glob pattern for checkpoint directories (default: checkpoint-*)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Resume the training run on wandb and log aggregated pass@K metrics",
    )
    args = parser.parse_args()

    # Load experiment config
    config_path = join(args.exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at {config_path}")

    with open(config_path) as f:
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

    # Create dataset to get blank_identifier_id
    eval_config = TRMDatasetConfig(
        test_set_mode=True,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        **exp_config["dataset"],
    )
    eval_dataset = TRMIterableDataset(eval_config, split="test")
    blank_identifier_id = eval_dataset.metadata["blank_identifier_id"]

    # Create evaluator for aggregation
    evaluator = ARC(
        data_path=data_path,
        blank_identifier_id=blank_identifier_id,
        submission_K=args.submission_k,
        pass_Ks=tuple(args.pass_ks),
        aggregated_voting=True,
    )

    # Scan all checkpoints
    ckpt_dirs = sorted(glob(join(args.exp_dir, args.ckpt_pattern)))
    logging.info(
        f"Found {len(ckpt_dirs)} checkpoint directories matching '{args.ckpt_pattern}'"
    )

    if not ckpt_dirs:
        logging.error(f"No checkpoints found in {args.exp_dir}")
        return

    # Load and aggregate all checkpoint states
    loaded_count = 0
    for ckpt_dir in ckpt_dirs:
        state_path = join(ckpt_dir, "arc_eval_state.pt")
        if not os.path.exists(state_path):
            logging.warning(f"Missing {state_path}, skipping")
            continue

        logging.info(f"Loading {state_path}")
        state = torch.load(state_path, weights_only=False)
        evaluator.merge_state_dict(state)
        loaded_count += 1

    if loaded_count == 0:
        logging.error(
            "No checkpoint states found! Run arc_eval.py on checkpoints first."
        )
        return

    logging.info(f"Aggregated {loaded_count} checkpoint states")

    # Add checkpoint count suffix to output_dir
    output_dir = args.output_dir or join(args.exp_dir, "arc_aggregate")
    output_dir = f"{output_dir}_ckpt{loaded_count}"
    os.makedirs(output_dir, exist_ok=True)

    # Compute aggregated results
    results = evaluator.result(save_path=output_dir)

    # Save results
    results_path = join(output_dir, "arc_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    logging.info("=" * 60)
    logging.info("Aggregated ARC Results:")
    logging.info("=" * 60)
    for metric_name, value in sorted(results.items()):
        logging.info(f"  {metric_name}: {value:.4f} ({value * 100:.2f}%)")
    logging.info("=" * 60)
    logging.info(f"Results saved to {results_path}")
    logging.info(f"Submission saved to {join(output_dir, 'submission.json')}")

    if args.wandb:
        run_name = exp_config.get("train", {}).get("run_name")
        if not run_name:
            logging.warning(
                "[ARC Aggregate] train.run_name missing from config, skipping wandb upload"
            )
        else:
            _log_aggregate_to_wandb(
                run_name=run_name,
                results=results,
                num_ckpts=loaded_count,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
