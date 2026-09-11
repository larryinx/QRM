import os

import torch.nn

KEY_MAPPING = {
    "train.per_device_train_batch_size": "train_bs",
    "train.per_device_eval_batch_size": "eval_bs",
    "train.gradient_accumulation_steps": "grad_accum",
    "train.learning_rate": "lr",
    "train.num_train_epochs": "epoch",
    "train.epochs": "epochs",
    "train.lr": "lr",
    "train.puzzle_emb_lr": "pelr",
    "model.use_fsq": "fsq",
    "model.fsq_levels": "flevels",
    "model.fsq_top_k": "ftopk",
    "model.fsq_sampling_training": "fsamp_tr",
    "model.fsq_sampling_inference": "fsamp_inf",
    "model.fsq_residual_mode": "fres_mode",
    "model.fsq_residual_weight": "fres_w",
    "model.fsq_recon_weight": "frecon_w",
    "model.fsq_pre_norm": "fpre",
    "model.num_iterations": "niter",
    "model.alpha": "alpha",
    "model.c_uct": "cuct",
    "model.weighted_em_loss": "wem",
    "model.weighted_em_weight": "wem_w",
}


def extract_key_params() -> str:
    """Extract all parameters from Hydra's override_dirname"""
    try:
        from hydra.core.hydra_config import HydraConfig

        hconf = HydraConfig.get()
        override_list = hconf.overrides.task
    except Exception:
        return ""

    if len(override_list) == 0:
        return ""

    parts = []
    for item in override_list:
        item_key = item.split("=")[0]
        item_value = item.split("=", 1)[1]
        if item_key in [
            "model",
            "dataset",
            "train",
            "train.output_dir",
            "train.dataloader_num_workers",
            "train.group_by_length",
            "train.gradient_checkpointing",
        ]:
            continue
        elif item_key in KEY_MAPPING:
            item_key = KEY_MAPPING[item_key]

        item_key = item_key.replace("+", "")
        item_value = item_value.strip("/").replace("/", "-")
        parts.append(f"{item_key}--{item_value}")

    return "__".join(parts)


def get_unique_job_id() -> str:
    job_id = ""
    # slurm case
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job_id = (
            f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        )
    # sge case
    if "JOB_ID" in os.environ:
        job_id = os.environ["JOB_ID"]
    if "SGE_TASK_ID" in os.environ and os.environ["SGE_TASK_ID"] != "undefined":
        job_id = f"{os.environ['JOB_ID']}_{os.environ['SGE_TASK_ID']}"
    # PJM case
    if "PJM_JOBID" in os.environ:
        job_id = os.environ["PJM_JOBID"]
    if "PBS_JOBID" in os.environ:
        job_id = os.environ["PBS_JOBID"]
    return job_id


def print_model_params(model: torch.nn.Module):
    """Print model parameters and buffers summary.

    Displays:
    - Parameters (nn.Parameter): Trainable and frozen
    - Buffers (nn.Buffer): Persistent and non-persistent
    - Total count combining both params and buffers
    """

    def format_params(num):
        """Format parameter count"""
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        else:
            return str(num)

    # ========== Parameters (nn.Parameter) ==========
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    param_groups = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        module_name = name.split(".")[0]

        if module_name not in param_groups:
            param_groups[module_name] = {"trainable": 0, "frozen": 0, "buffer": 0}

        if param.requires_grad:
            trainable_params += num_params
            param_groups[module_name]["trainable"] += num_params
        else:
            frozen_params += num_params
            param_groups[module_name]["frozen"] += num_params

    # ========== Buffers (nn.Buffer) ==========
    total_buffers = 0
    buffer_groups = {}

    for name, buffer in model.named_buffers():
        num_elements = buffer.numel()
        total_buffers += num_elements
        module_name = name.split(".")[0]

        if module_name not in param_groups:
            param_groups[module_name] = {"trainable": 0, "frozen": 0, "buffer": 0}
        param_groups[module_name]["buffer"] += num_elements

        if module_name not in buffer_groups:
            buffer_groups[module_name] = 0
        buffer_groups[module_name] += num_elements

    # ========== Summary ==========
    grand_total = total_params + total_buffers

    print("\n" + "=" * 70)
    print("MODEL SIZE SUMMARY")
    print("=" * 70)

    print("\n[Parameters (nn.Parameter)]")
    print(
        f"  Total Parameters:     {format_params(total_params):>12} ({total_params:,})"
    )
    print(
        f"  Trainable Parameters: {format_params(trainable_params):>12} ({trainable_params:,})"
    )
    print(
        f"  Frozen Parameters:    {format_params(frozen_params):>12} ({frozen_params:,})"
    )
    if total_params > 0:
        print(
            f"  Trainable Ratio:      {trainable_params / total_params * 100:>11.2f}%"
        )

    print("\n[Buffers (nn.Buffer)]")
    print(
        f"  Total Buffers:        {format_params(total_buffers):>12} ({total_buffers:,})"
    )

    print("\n[Grand Total]")
    print(f"  Params + Buffers:     {format_params(grand_total):>12} ({grand_total:,})")

    # ========== Breakdown by Module ==========
    print("\n" + "-" * 70)
    print("BREAKDOWN BY MODULE")
    print("-" * 70)
    print(
        f"{'Module':<25} {'Trainable':<12} {'Frozen':<12} {'Buffer':<12} {'Total':<12}"
    )
    print("-" * 70)

    for module_name in sorted(param_groups.keys()):
        train_p = param_groups[module_name]["trainable"]
        frozen_p = param_groups[module_name]["frozen"]
        buffer_p = param_groups[module_name]["buffer"]
        total_p = train_p + frozen_p + buffer_p
        print(
            f"{module_name:<25} {format_params(train_p):<12} {format_params(frozen_p):<12} "
            f"{format_params(buffer_p):<12} {format_params(total_p):<12}"
        )

    # Print totals row
    print("-" * 70)
    print(
        f"{'TOTAL':<25} {format_params(trainable_params):<12} {format_params(frozen_params):<12} "
        f"{format_params(total_buffers):<12} {format_params(grand_total):<12}"
    )
