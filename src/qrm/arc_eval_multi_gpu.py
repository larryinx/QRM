"""ARC Multi-GPU Evaluation Script - Distributed inference on a single checkpoint.

Same semantics as arc_eval.py but each rank processes a disjoint shard of the test
set (TRMIterableDataset already splits batches by rank/num_replicas). After inference
each rank saves its partial evaluator state; rank 0 merges shards into the single
`arc_eval_state.pt` file the aggregate script expects, writes metrics/submission,
and optionally logs metrics to wandb (resuming the training run).

Usage:
    torchrun --nproc_per_node=4 -m qrm.arc_eval_multi_gpu \
        --ckpt_dir results/arc_agi/run_xxx/checkpoint-10000 [--wandb]

Rank 0 outputs (unchanged from arc_eval.py):
    {ckpt_dir}/arc_eval_state.pt
    {ckpt_dir}/arc_metrics.json
    {ckpt_dir}/arc_results/submission.json
"""

import argparse
import json
import logging
import os
import re
from os.path import join

import torch
import torch.distributed as dist
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
    assert (
        len(features) == 1
    ), "IterableDataset with batch_size=1 should yield one batch at a time"
    return features[0]


def run_arc_evaluation_rank(
    trainer: TRMTrainer, evaluator: ARC, device: torch.device, is_rank0: bool
):
    """Per-rank ARC evaluation loop (mirror of arc_eval.run_arc_evaluation).

    Each rank runs its own inference on its shard of the test set. EMA swap, carry
    management, and the halt-loop are identical to the single-GPU path.
    """
    # Build the DataLoader directly instead of trainer.get_eval_dataloader().
    # TRMIterableDataset already shards via rank/num_replicas; HF Trainer would
    # additionally wrap it with IterableDatasetShard, causing double-sharding
    # (each rank seeing only 1/world_size of its already-sharded batches).
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        trainer.eval_dataset, batch_size=1, collate_fn=passthrough_collator
    )

    model = trainer.model
    if hasattr(model, "module"):
        model = model.module

    if trainer.use_ema and trainer.ema_helper is not None:
        if trainer.eval_use_compile:
            trainer.ema_helper.swap_weights(model)
            if is_rank0:
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
            if is_rank0:
                logging.info(
                    f"[ARC Eval] Using EMA weights (mu={trainer.ema_helper.mu})"
                )
                log_weight_comparison(
                    params_a={_clean_param_name(n): p for n, p in model.named_parameters()},
                    params_b=trainer.ema_helper.shadow,
                    label_a="Eval (EMA)",
                    label_b="Shadow (EMA)",
                    prefix="[ARC Eval]",
                )
    elif is_rank0:
        logging.info("[ARC Eval] Using original model for evaluation (no EMA)")

    model.eval()
    evaluator.begin_eval()
    processed_batches = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="ARC Eval", disable=not is_rank0):
            processed_batches += 1
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
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

    logging.info(f"[rank{dist.get_rank() if dist.is_initialized() else 0}] Processed {processed_batches} batches")


def _parse_ckpt_step(ckpt_dir: str):
    """Return (step, label) usable as a wandb x-axis for this checkpoint.

    - checkpoint-{N}  -> (N, "checkpoint-N")
    - final_checkpoint with trainer_state.json -> (global_step, "final")
    - fallback -> (0, basename)
    """
    name = os.path.basename(ckpt_dir.rstrip("/"))
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        return int(m.group(1)), name
    trainer_state_path = join(ckpt_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path) as f:
                state = json.load(f)
            step = int(state.get("global_step", 0))
            return step, name
        except Exception:
            pass
    return 0, name


def _build_wandb_id(run_name: str) -> str:
    """Mirror the id formula used by qrm.train:65-69 so we resume the same run."""
    existing = os.environ.get("WANDB_RUN_ID")
    if existing:
        return existing
    project = os.environ.get("WANDB_PROJECT", "")
    raw = f"{project}__{run_name}" if project else run_name
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:128]


def _log_single_to_wandb(run_name, metrics, ckpt_step, ckpt_label, ckpt_dir):
    """Resume the training run and log per-checkpoint ARC metrics on a decoupled axis."""
    try:
        import wandb
    except ImportError:
        logging.warning("[ARC Eval] wandb not installed, skipping upload")
        return

    project = os.environ.get("WANDB_PROJECT")
    if not project:
        logging.warning("[ARC Eval] WANDB_PROJECT not set, skipping wandb upload")
        return

    run_id = _build_wandb_id(run_name)
    logging.info(
        f"[ARC Eval] Resuming wandb run {project}/{run_id} (step={ckpt_step}, label={ckpt_label})"
    )

    try:
        run = wandb.init(project=project, id=run_id, resume="must", reinit=True)
    except Exception as e:
        logging.warning(
            f"[ARC Eval] Resume failed for {project}/{run_id}: {e}. "
            f"Creating a new run with the same id."
        )
        run = wandb.init(project=project, id=run_id, resume="allow", reinit=True)

    # Decoupled x-axis so arc metrics don't collide with training's _step.
    wandb.define_metric("arc_eval/ckpt_step")
    wandb.define_metric("arc_eval/*", step_metric="arc_eval/ckpt_step")

    log_dict = {}
    for k, v in metrics.items():
        # results keys look like "ARC/pass@1"; map to "arc_eval/pass@1"
        suffix = k.split("/", 1)[1] if "/" in k else k
        log_dict[f"arc_eval/{suffix}"] = v
    log_dict["arc_eval/ckpt_step"] = ckpt_step

    wandb.log(log_dict)

    # Mirror latest numbers into summary for at-a-glance visibility.
    for k, v in metrics.items():
        suffix = k.split("/", 1)[1] if "/" in k else k
        run.summary[f"arc_latest/{suffix}"] = v
    run.summary["arc_latest/ckpt_step"] = ckpt_step
    run.summary["arc_latest/ckpt_label"] = ckpt_label
    run.summary["arc_latest/ckpt_dir"] = os.path.abspath(ckpt_dir)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU ARC evaluation for a single checkpoint"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--eval_use_compile", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Resume the training run on wandb and log per-checkpoint pass@K (rank 0 only)",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = rank == 0
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    output_path = join(args.ckpt_dir, "arc_eval_state.pt")
    should_skip = os.path.exists(output_path) and not args.force
    if is_distributed:
        skip_tensor = torch.tensor([int(should_skip)], device=device)
        dist.broadcast(skip_tensor, src=0)
        should_skip = bool(skip_tensor.item())
    if should_skip:
        if is_rank0:
            logging.info(
                f"Skipping {args.ckpt_dir} (already evaluated, use --force to override)"
            )
        if is_distributed:
            dist.destroy_process_group()
        return

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
    if is_rank0:
        logging.info(f"Using data path: {data_path}")

    identifiers_path = join(data_path, "identifiers.json")
    test_puzzles_path = join(data_path, "test_puzzles.json")
    if not os.path.exists(identifiers_path):
        raise FileNotFoundError(f"identifiers.json not found at {identifiers_path}")
    if not os.path.exists(test_puzzles_path):
        raise FileNotFoundError(f"test_puzzles.json not found at {test_puzzles_path}")

    eval_config = TRMDatasetConfig(
        test_set_mode=True,
        rank=rank,
        num_replicas=world_size,
        epochs_per_iter=1,
        **exp_config["dataset"],
    )
    eval_dataset = TRMIterableDataset(eval_config, split="test")

    config = TRMConfig(
        batch_size=exp_config["dataset"]["global_batch_size"] // world_size,
        seq_len=eval_dataset.metadata["seq_len"],
        vocab_size=eval_dataset.metadata["vocab_size"],
        num_puzzle_identifiers=eval_dataset.metadata["num_puzzle_identifiers"],
        **exp_config["model"],
    )

    with torch.device(device):
        model = TRMForPuzzleSolving(config)

    training_args = torch.load(f"{args.ckpt_dir}/training_args.bin", weights_only=False)
    training_args.report_to = "none"
    training_args.torch_compile = True
    training_args.torch_compile_backend = "inductor"

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

    # HF Trainer.__init__ calls _move_model_to_device(model, args.device) and
    # TrainingArguments.device is a cached_property. A pickled training_args.bin
    # can carry a stale cache that resolves to cuda:0 on every rank, which silently
    # moves the model off this rank's device. Force it back, along with the EMA
    # shadow (loaded to next(model.parameters()).device inside _load_trm_extra_state).
    trainer.model.to(device)
    if trainer.ema_helper is not None:
        trainer.ema_helper.shadow = {
            k: v.to(device) for k, v in trainer.ema_helper.shadow.items()
        }

    # Per-rank evaluator. aggregated_voting=False because each rank's begin_eval()
    # should leave state empty (no cross-evaluation accumulation on a single ckpt).
    evaluator = ARC(
        data_path=data_path,
        blank_identifier_id=eval_dataset.metadata["blank_identifier_id"],
        aggregated_voting=False,
    )

    if is_rank0:
        logging.info(
            f"Running ARC evaluation on {args.ckpt_dir} (world_size={world_size})"
        )
    run_arc_evaluation_rank(trainer, evaluator, device, is_rank0=is_rank0)

    shard_path = join(args.ckpt_dir, f"arc_eval_state_rank{rank}.pt")
    torch.save(evaluator.state_dict(), shard_path)
    if is_distributed:
        dist.barrier()

    if is_rank0:
        merged = ARC(
            data_path=data_path,
            blank_identifier_id=eval_dataset.metadata["blank_identifier_id"],
            aggregated_voting=False,
        )
        for r in range(world_size):
            shard = join(args.ckpt_dir, f"arc_eval_state_rank{r}.pt")
            merged.merge_state_dict(torch.load(shard, weights_only=False))
        torch.save(merged.state_dict(), output_path)
        logging.info(
            f"Merged {world_size} rank shards -> {output_path}"
        )

        for r in range(world_size):
            shard = join(args.ckpt_dir, f"arc_eval_state_rank{r}.pt")
            try:
                os.remove(shard)
            except FileNotFoundError:
                pass

        results_dir = join(args.ckpt_dir, "arc_results")
        results = merged.result(save_path=results_dir)
        logging.info(f"Single checkpoint results: {results}")

        metrics_path = join(args.ckpt_dir, "arc_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")

        if args.wandb:
            run_name = getattr(training_args, "run_name", None)
            if not run_name:
                logging.warning(
                    "[ARC Eval] training_args.run_name is empty, skipping wandb upload"
                )
            else:
                ckpt_step, ckpt_label = _parse_ckpt_step(args.ckpt_dir)
                _log_single_to_wandb(
                    run_name=run_name,
                    metrics=results,
                    ckpt_step=ckpt_step,
                    ckpt_label=ckpt_label,
                    ckpt_dir=args.ckpt_dir,
                )

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
