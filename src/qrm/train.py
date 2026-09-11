import glob
import os
import sys

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from qrm import PROJECT_ROOT
from qrm.data.puzzle_dataset import TRMDatasetConfig, TRMIterableDataset
from qrm.models.trm.configuration_trm import TRMConfig
from qrm.models.trm.modeling_trm import TRMForPuzzleSolving as TRMModel
from qrm.models.qrm.configuration_qrm import QRMConfig
from qrm.models.qrm.modeling_qrm import QRMForPuzzleSolving as QRMModel
from qrm.trainers.trm import TRMTrainer
from qrm.trainers.qrm import QRMTrainer
from qrm.utils.train import extract_key_params, get_unique_job_id, print_model_params

# Registry: model_type -> (ConfigClass, ModelClass, TrainerClass)
MODEL_REGISTRY = {
    "trm": (TRMConfig, TRMModel, TRMTrainer),
    "qrm": (QRMConfig, QRMModel, QRMTrainer),
}


@hydra.main(config_path=f"{PROJECT_ROOT}/config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    cfg: Contains all user configurations (base.yaml + arch/*.yaml + command line overrides)
         Does not include hydra.* configurations
    """
    from hydra.core.hydra_config import HydraConfig

    hconf = HydraConfig.get()

    # # Disable MPI for DeepSpeed (use torch.distributed instead)
    # os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "true")
    # os.environ.setdefault("DEEPSPEED_COMM_BACKEND", "nccl")

    # Disable tokenizers parallelism warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ============ Distributed training info ============
    rank = int(os.environ.get("RANK", 0))  # Global rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    job_id = get_unique_job_id()
    override_params_str = extract_key_params()

    run_name = f"{hconf.runtime.choices.model}__{hconf.runtime.choices.dataset}__{hconf.runtime.choices.train}"
    if job_id:
        run_name = f"{job_id}__{run_name}"
    if override_params_str:
        run_name += f"__{override_params_str}"
    run_name = run_name.replace("+", "")

    # FIXME: Build a meaningful wandb run ID so the local wandb/ folders read as
    #   run-YYYYMMDD_HHMMSS-QRM_MS16_ARC__trm_att_arc_agi__arc_agi_1__...
    # instead of
    #   run-YYYYMMDD_HHMMSS-r9k2py1v
    # _wandb_project = os.environ.get("WANDB_PROJECT", "")
    # _wandb_id = f"{_wandb_project}__{run_name}" if _wandb_project else run_name
    # _wandb_id = "".join(
    #     c if c.isalnum() or c in "-_." else "_" for c in _wandb_id
    # )[:128]
    # os.environ.setdefault("WANDB_RUN_ID", _wandb_id)

    output_dir = f"{PROJECT_ROOT}/results/{run_name}"

    cfg.train.run_name = run_name
    if getattr(cfg.train, "output_dir", None) is None:
        cfg.train.output_dir = output_dir
    else:
        output_dir = cfg.train.output_dir
        # TODO: add logics to automatically read configuration from output_dir

    train_config = TRMDatasetConfig(
        test_set_mode=False,
        rank=rank,
        num_replicas=world_size,
        epochs_per_iter=cfg.train.eval_interval,
        **cfg.dataset,
    )
    train_dataset = TRMIterableDataset(train_config, split="train")

    eval_config = TRMDatasetConfig(
        test_set_mode=True,
        rank=rank,
        num_replicas=world_size,
        epochs_per_iter=1,
        **cfg.dataset,
    )
    eval_dataset = TRMIterableDataset(eval_config, split="test")

    # Derive model type from Hydra config name: "qrm_att_sudoku" → "qrm"
    model_type = hconf.runtime.choices.model.split("_")[0]
    assert model_type in MODEL_REGISTRY, (
        f"Unknown model type '{model_type}' from config '{hconf.runtime.choices.model}'. "
        f"Expected one of: {list(MODEL_REGISTRY.keys())}"
    )
    # Persist model_type in config so test.py can read it back
    OmegaConf.update(cfg, "model_type", model_type, force_add=True)

    # QRM: gradient accumulation spans one MCTS tree search
    # Each training_step() = one MCTS iteration, num_iterations per optimizer update.
    # train.gradient_accumulation_steps overrides this when > 0 (decouples optimizer
    # frequency from num_iterations — e.g. set to 1 with fsq_top_k=1 and
    # num_iterations=halt_max_steps to reproduce TRM+FSQ behavior).
    gradient_accumulation_steps = 1
    use_torch_compile = True
    if model_type == "qrm":
        explicit_grad_accum = cfg.train.get("gradient_accumulation_steps", 0)
        if explicit_grad_accum > 0:
            gradient_accumulation_steps = explicit_grad_accum
        else:
            gradient_accumulation_steps = cfg.model.get("num_iterations", 1)

    steps_per_epoch = (
        train_dataset.metadata["total_groups"]
        * train_dataset.metadata["mean_puzzle_examples"]
        / cfg.dataset.global_batch_size
    )
    # HF Trainer's max_steps counts optimizer updates.
    # With gradient_accumulation_steps=N, each update consumes N training_step() calls.
    # Dividing keeps the same total data consumption (epochs) as TRM.
    total_steps = int(cfg.train.epochs * steps_per_epoch) // gradient_accumulation_steps
    eval_steps = int(cfg.train.eval_interval * steps_per_epoch) // gradient_accumulation_steps
    # Ensure at least 1 step
    total_steps = max(total_steps, 1)
    eval_steps = max(eval_steps, 1)

    training_args = TrainingArguments(
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        seed=cfg.dataset.seed,
        max_steps=total_steps,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        torch_compile=use_torch_compile,
        torch_compile_backend="inductor",
        torch_compile_mode=None,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=8,  # may not be used by the Trainer backend
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        report_to=["wandb"],
        run_name=cfg.train.run_name,
        output_dir=cfg.train.output_dir,
        resume_from_checkpoint=True,
        save_only_model=False,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        adam_beta1=cfg.train.beta1,
        adam_beta2=cfg.train.beta2,
        warmup_steps=cfg.train.lr_warmup_steps,  # Check consistency
        # temporary set to full precision for reproducibility
        bf16=False,
        fp16=False,
        tf32=False,
    )

    # Model selection via registry
    ConfigClass, ModelClass, TrainerClass = MODEL_REGISTRY[model_type]

    config = ConfigClass(
        batch_size=cfg.dataset.global_batch_size // world_size,
        seq_len=train_dataset.metadata["seq_len"],
        vocab_size=train_dataset.metadata["vocab_size"],
        num_puzzle_identifiers=train_dataset.metadata["num_puzzle_identifiers"],
        **cfg.model,
    )

    # Use local_rank to specify CUDA device
    # torchrun sets LOCAL_RANK environment variable, each process maps to one GPU
    # Use torch.device context to create model directly on GPU, ensuring buffers are leaf tensors
    # (avoid .to(device) causing requires_grad=True buffers to become non-leaf)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    with torch.device(device):
        model = ModelClass(config)

    # Print model parameters info
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nTrainingArguments output_dir: {output_dir}")

        print("=" * 60)
        print("Hydra Runtime Info")
        print("=" * 60)
        print(f"Job name: {hconf.job.name}")
        print(f"Output dir: {output_dir}")
        print(f"Working dir: {hconf.runtime.cwd}")
        print(
            f"Distributed: {is_distributed} (World size: {world_size}, Rank: {rank}, Local rank: {local_rank})"
        )
        print(f"Selected model: {hconf.runtime.choices.model}")
        print(f"Command line overrides: {hconf.overrides.task}")

        print("\n" + "=" * 60)
        print("User Configuration (Merged base + arch)")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

        print("\n" + "=" * 60)
        print("Train Dataset Config")
        print("=" * 60)
        print(train_dataset.config)

        print("\n" + "=" * 60)
        print("Eval Dataset Config")
        print("=" * 60)
        print(eval_dataset.config)

        print("\n" + "=" * 60)
        print("Model Config")
        print("=" * 60)
        print(model.config)

        print("\n" + "=" * 60)
        print("Model Parameters Summary")
        print("=" * 60)
        print_model_params(model)
        print("=" * 60 + "\n")

        # Save complete config
        # Use resolve=True to resolve all interpolations (e.g., ${.hidden_size}, ${oc.env:PWD})
        config_save_path = f"{output_dir}/config.yaml"
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f, resolve=True)
        print(f"\nConfig saved to: {config_save_path}")

        # Note: TrainingArguments will be auto-saved by Trainer as training_args.bin
        # We save a simplified version for human readability
        training_args_save_path = f"{output_dir}/training_args.yaml"
        try:
            with open(training_args_save_path, "w") as f:
                # Only save serializable attributes
                safe_dict = training_args.to_dict()
                yaml.dump(safe_dict, f, default_flow_style=False)
            print(f"TrainingArguments saved to {training_args_save_path}")
        except Exception as e:
            print(f"Warning: Could not save training args to YAML: {e}")
            print(
                "TrainingArguments will still be auto-saved by Trainer as training_args.bin"
            )

    # Custom data_collator: passthrough data directly
    # TRMIterableDataset handles batching internally, returns complete batch dict
    # Default collator would try to stack, causing dimension errors
    def passthrough_collator(features):
        assert (
            len(features) == 1
        ), "IterableDataset with batch_size=1 should yield one batch at a time"
        return features[0]

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=passthrough_collator,
        puzzle_emb_lr=cfg.train.puzzle_emb_lr,
        puzzle_emb_weight_decay=cfg.train.puzzle_emb_weight_decay,
        lr_min_ratio=cfg.train.lr_min_ratio,
        ema=cfg.train.ema,
        ema_rate=cfg.train.ema_rate,
        eval_use_compile=cfg.train.eval_use_compile,
    )

    trainer = TrainerClass(**trainer_kwargs)

    ckpt_list = sorted(glob.glob(f"{glob.escape(training_args.output_dir)}/checkpoint-*"))
    resume_from_checkpoint = len(ckpt_list) >= 1
    if resume_from_checkpoint:
        print(f"[INFO] Resuming training from checkpoint in {training_args.output_dir}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final checkpoint (includes model, EMA, SignSGD, Dataset state)
    # Note: trainer.save_model() only saves model weights, not EMA etc.
    final_checkpoint_dir = f"{training_args.output_dir}/final_checkpoint"
    trainer.save_final_checkpoint(output_dir=final_checkpoint_dir)
    if rank == 0:
        print(f"[INFO] Final checkpoint saved to {final_checkpoint_dir}")


if __name__ == "__main__":
    # Filter out DeepSpeed launcher arguments that Hydra doesn't recognize
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]
    main()
