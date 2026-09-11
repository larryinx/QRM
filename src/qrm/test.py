import argparse
import json
import logging
import os
from os.path import join

import torch
import yaml

from qrm.data.puzzle_dataset import TRMDatasetConfig, TRMIterableDataset
from qrm.models.trm.configuration_trm import TRMConfig
from qrm.models.trm.modeling_trm import TRMForPuzzleSolving as TRMModel
from qrm.models.qrm.configuration_qrm import QRMConfig
from qrm.models.qrm.modeling_qrm import QRMForPuzzleSolving as QRMModel
from qrm.trainers.trm import TRMTrainer
from qrm.trainers.qrm import QRMTrainer

# Registry: model_type -> (ConfigClass, ModelClass, TrainerClass)
MODEL_REGISTRY = {
    "trm": (TRMConfig, TRMModel, TRMTrainer),
    "qrm": (QRMConfig, QRMModel, QRMTrainer),
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--eval_use_compile", action="store_true")
    args = parser.parse_args()

    test_metrics_path = join(args.ckpt_dir, "test_metrics.json")
    if args.eval_use_compile:
        test_metrics_path = test_metrics_path.replace(".json", "_compile.json")
    if os.path.exists(test_metrics_path):
        logging.info(f"Skipping {args.ckpt_dir} (already tested)")
        return

    exp_config_path = join(args.ckpt_dir, "../config.yaml")
    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    # Read model type saved by train.py
    model_type = exp_config.get("model_type", None)
    if model_type is None:
        # Fallback for old checkpoints that don't have model_type saved
        model_type = "qrm" if exp_config.get("model", {}).get("use_fsq", False) else "trm"
        logging.warning(f"model_type not found in config, inferred '{model_type}' from use_fsq")

    eval_config = TRMDatasetConfig(
        test_set_mode=True,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        **exp_config["dataset"],
    )
    eval_dataset = TRMIterableDataset(eval_config, split="test")

    ConfigClass, ModelClass, TrainerClass = MODEL_REGISTRY[model_type]

    config = ConfigClass(
        batch_size=exp_config["dataset"]["global_batch_size"],
        seq_len=eval_dataset.metadata["seq_len"],
        vocab_size=eval_dataset.metadata["vocab_size"],
        num_puzzle_identifiers=eval_dataset.metadata["num_puzzle_identifiers"],
        **exp_config["model"],
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with torch.device(device):
        model = ModelClass(config)

    training_args = torch.load(f"{args.ckpt_dir}/training_args.bin", weights_only=False)
    training_args.report_to = "none"
    training_args.torch_compile = True
    training_args.torch_compile_backend = "inductor"

    def passthrough_collator(features):
        assert (
            len(features) == 1
        ), "IterableDataset with batch_size=1 should yield one batch at a time"
        return features[0]

    trainer_kwargs = dict(
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

    trainer = TrainerClass(**trainer_kwargs)

    trainer._load_from_checkpoint(args.ckpt_dir)
    metrics = trainer.evaluate()

    with open(test_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
