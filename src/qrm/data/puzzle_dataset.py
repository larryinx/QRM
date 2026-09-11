"""TRM Puzzle Dataset for training and evaluation."""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from qrm.losses.trm import IGNORE_LABEL_ID


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Pack examples into a full batch."""
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(
            puzzle_start + np.random.choice(puzzle_size, append_size, replace=False)
        )

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class TRMDatasetConfig:
    """Dataset configuration."""

    def __init__(
        self,
        seed: int,
        dataset_paths: List[str],
        global_batch_size: int,
        test_set_mode: bool,
        epochs_per_iter: int,  # Batch X epochs in an iteration to reduce overhead.
        rank: int,
        num_replicas: int,
    ):
        self.seed = seed
        self.dataset_paths = dataset_paths
        self.global_batch_size = global_batch_size
        self.test_set_mode = test_set_mode
        self.epochs_per_iter = epochs_per_iter
        self.rank = rank
        self.num_replicas = num_replicas

    def __repr__(self) -> str:
        attrs = ",\n  ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"TRMDatasetConfig(\n  {attrs}\n)"


class TRMIterableDataset(IterableDataset):
    """TRM Dataset with group-based sampling and distributed support.

    Key features:
    1. Group-based sampling
    2. Internal batch assembly
    3. Distributed rank partitioning
    """

    def __init__(self, config: TRMDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge metadata from multiple datasets
        self._load_and_merge_metadata()

        # Check batch size
        assert self.config.global_batch_size % self.config.num_replicas == 0
        self.local_batch_size = (
            self.config.global_batch_size // self.config.num_replicas
        )

        # State
        self._data = None
        self._iters = 0

    def state_dict(self) -> Dict:
        """Return dataset state for checkpoint saving.

        Used for training resume. _iters controls RNG seed, ensuring reproducible sampling.

        Usage:
        1. With torchdata.StatefulDataLoader (recommended):
           from torchdata.stateful_dataloader import StatefulDataLoader
           dataloader = StatefulDataLoader(dataset, batch_size=1)
           state = dataloader.state_dict()  # auto-calls dataset.state_dict()
           dataloader.load_state_dict(state)

        2. Manual save/load in custom Trainer:
           # Save
           dataset_state = trainer.train_dataset.state_dict()
           torch.save(dataset_state, "checkpoint/dataset_state.pt")
           # Load
           dataset_state = torch.load("checkpoint/dataset_state.pt")
           trainer.train_dataset.load_state_dict(dataset_state)
        """
        return {"_iters": self._iters}

    def load_state_dict(self, state_dict: Dict):
        # Since all ranks use the same seed at init, _iters is shared across ranks
        self._iters = state_dict.get("_iters", 0)

    def __len__(self) -> int:
        """Return number of batches in dataset (for tqdm progress display).

        Note: IterableDataset typically doesn't implement __len__, but for progress bar support,
        we compute an estimated batch count based on mode.

        - test_set_mode: Exact calculation (total_examples / global_batch_size rounded up)
        - train_mode: Estimated value (based on epochs_per_iter and mean_puzzle_examples)
        """
        self._lazy_load_dataset()

        if self.config.test_set_mode:
            # Test mode: exact calculation
            total_batches = 0
            for set_name, dataset in self._data.items():
                total_examples = len(dataset["inputs"])
                # Round up
                total_batches += (
                    total_examples + self.config.global_batch_size - 1
                ) // self.config.global_batch_size
            return total_batches
        else:
            # Train mode: estimated value
            total_batches = 0
            for set_name, dataset in self._data.items():
                num_groups = dataset["group_indices"].size - 1
                # epochs_per_iter epochs, each iterating all groups
                total_groups = num_groups * self.config.epochs_per_iter
                # Each group produces mean_puzzle_examples samples on average
                # But due to group-based sampling, this is just an estimate
                estimated_samples = total_groups * self.metadata["mean_puzzle_examples"]
                total_batches += int(estimated_samples / self.config.global_batch_size)
            return total_batches

    def _load_and_merge_metadata(self):
        """Merge metadata from multiple datasets."""
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0

        for dataset_path in self.config.dataset_paths:
            with open(os.path.join(dataset_path, self.split, "dataset.json")) as f:
                current_metadata = json.load(f)

            if prev_seq_len is None:
                prev_seq_len = current_metadata["seq_len"]
                prev_vocab_size = current_metadata["vocab_size"]
                prev_pad_id = current_metadata["pad_id"]
                prev_ignore_label_id = current_metadata.get("ignore_label_id")
                prev_blank_identifier_id = current_metadata["blank_identifier_id"]
                prev_sets = current_metadata["sets"]
                prev_num_identifiers = current_metadata["num_puzzle_identifiers"]
            else:
                assert prev_seq_len == current_metadata["seq_len"]
                assert prev_vocab_size == current_metadata["vocab_size"]
                assert prev_pad_id == current_metadata["pad_id"]
                assert prev_ignore_label_id == current_metadata.get("ignore_label_id")
                assert (
                    prev_blank_identifier_id == current_metadata["blank_identifier_id"]
                )
                assert prev_sets == current_metadata["sets"]
                assert (
                    prev_num_identifiers == current_metadata["num_puzzle_identifiers"]
                )

            mean_puzzle_examples += (
                current_metadata["mean_puzzle_examples"]
                * current_metadata["total_puzzles"]
            )
            total_puzzles += current_metadata["total_puzzles"]
            total_groups += current_metadata["total_groups"]
            num_identifiers += current_metadata["num_puzzle_identifiers"]

        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = {
            "seq_len": prev_seq_len,
            "vocab_size": prev_vocab_size,
            "pad_id": prev_pad_id,
            "ignore_label_id": prev_ignore_label_id,
            "blank_identifier_id": prev_blank_identifier_id,
            "num_puzzle_identifiers": num_identifiers,
            "total_groups": total_groups,
            "mean_puzzle_examples": mean_puzzle_examples,
            "total_puzzles": total_puzzles,
            "sets": prev_sets,
        }

    def _lazy_load_dataset(self):
        """Lazy load dataset."""
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }

        # Optional fields (for backward compatibility)
        optional_fields = {
            "puzzle_aug_strings": None,  # Full augmentation strings for inverse_aug
        }

        # Load data
        self._data = {}
        for set_name in self.metadata["sets"]:  # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                set_name_ = set_name + str(i) if i > 0 else set_name
                self._data[set_name_] = {
                    field_name: np.load(
                        os.path.join(
                            dataset_path, self.split, f"{set_name}__{field_name}.npy"
                        ),
                        mmap_mode=mmap_mode,
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

                # Load optional fields if they exist
                for field_name, mmap_mode in optional_fields.items():
                    optional_path = os.path.join(
                        dataset_path, self.split, f"{set_name}__{field_name}.npy"
                    )
                    if os.path.exists(optional_path):
                        self._data[set_name_][field_name] = np.load(
                            optional_path,
                            mmap_mode=mmap_mode,
                            allow_pickle=True,  # Required for object arrays (strings)
                        )

    def _collate_batch(
        self, batch: Dict[str, np.ndarray], string_batch: Dict[str, np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """Collate batch.

        Args:
            batch: Dict of numeric arrays to convert to tensors.
            string_batch: Optional dict of string arrays (not converted to tensors).

        Returns:
            Dict of tensors and optionally string arrays.
        """
        # Convert dtype for numeric arrays
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata["ignore_label_id"] is not None:
            batch["labels"][
                batch["labels"] == self.metadata["ignore_label_id"]
            ] = IGNORE_LABEL_ID

        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata["pad_id"],
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata["blank_identifier_id"],
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k],
                )
                for k, v in batch.items()
            }

            # Pad string arrays with empty strings
            if string_batch:
                string_batch = {
                    k: np.pad(v, (0, pad_size), constant_values="")
                    for k, v in string_batch.items()
                }

        result = {k: torch.from_numpy(v) for k, v in batch.items()}
        if "inputs" in result:
            result["input_ids"] = result.pop("inputs")

        # Add string arrays directly (not as tensors)
        if string_batch:
            result.update(string_batch)

        return result

    def _iter_test(self):
        """Test set iteration."""
        for set_i, (set_name, dataset) in enumerate(self._data.items()):
            total_examples = len(dataset["inputs"])
            start_index = 0
            has_aug_strings = "puzzle_aug_strings" in dataset

            while start_index < total_examples:
                end_index = min(
                    total_examples, start_index + self.config.global_batch_size
                )

                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(
                    start_index + (self.config.rank + 1) * self.local_batch_size,
                    end_index,
                )

                puzzle_indices = []
                puzzle_index = (
                    np.searchsorted(
                        dataset["puzzle_indices"], local_start, side="right"
                    )
                    - 1
                )
                for i in range(local_start, local_end):
                    while (
                        puzzle_index + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_index + 1]
                    ):
                        puzzle_index += 1
                    puzzle_indices.append(puzzle_index)

                # Prepare string batch if available
                string_batch = None
                if has_aug_strings:
                    string_batch = {
                        "puzzle_aug_strings": dataset["puzzle_aug_strings"][
                            puzzle_indices
                        ]
                    }

                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][local_start:local_end],
                        "labels": dataset["labels"][local_start:local_end],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            puzzle_indices
                        ],
                    },
                    string_batch=string_batch,
                )

                yield batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        """Training set iteration."""
        for set_name, dataset in self._data.items():
            # Increase epoch count
            self._iters += 1
            has_aug_strings = "puzzle_aug_strings" in dataset

            # Randomly shuffle groups
            rng = np.random.Generator(
                np.random.Philox(seed=self.config.seed + self._iters)
            )

            group_order = np.concatenate(
                [
                    rng.permutation(dataset["group_indices"].size - 1)
                    for _i in range(self.config.epochs_per_iter)
                ]
            )
            start_index = 0

            while start_index < group_order.size:
                # Sample batch
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = (
                    batch_puzzle_indices.size
                )  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                # Slice for current rank
                batch_rank_start = self.config.rank * self.local_batch_size
                batch_rank_end = (self.config.rank + 1) * self.local_batch_size
                batch_indices = batch_indices[batch_rank_start:batch_rank_end]
                batch_puzzle_indices = batch_puzzle_indices[
                    batch_rank_start:batch_rank_end
                ]

                # Prepare string batch if available
                string_batch = None
                if has_aug_strings:
                    string_batch = {
                        "puzzle_aug_strings": dataset["puzzle_aug_strings"][
                            batch_puzzle_indices
                        ]
                    }

                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][batch_indices],
                        "labels": dataset["labels"][batch_indices],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            batch_puzzle_indices
                        ],
                    },
                    string_batch=string_batch,
                )

                yield batch

    def __iter__(self):
        """Iterator."""
        worker_info = get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


if __name__ == "__main__":
    # Sudoku-Extreme test configuration
    # Dataset path: data/sudoku-extreme-1k-aug-1000
    # Generate with: python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

    sudoku_dataset_path = "data/sudoku-extreme-1k-aug-1000"

    # Train config
    train_config = TRMDatasetConfig(
        seed=0,
        dataset_paths=[sudoku_dataset_path],
        global_batch_size=192,  # Default batch size (single GPU)
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    # Test config
    test_config = TRMDatasetConfig(
        seed=0,
        dataset_paths=[sudoku_dataset_path],
        global_batch_size=192,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    # Create datasets
    print("=" * 60)
    print("Testing Sudoku-Extreme Dataset")
    print("=" * 60)

    train_dataset = TRMIterableDataset(train_config, split="train")
    test_dataset = TRMIterableDataset(test_config, split="test")

    # Print metadata
    print("\n[Metadata]")
    for k, v in train_dataset.metadata.items():
        print(f"  {k}: {v}")

    # Test train set iteration
    print("\n[Train Dataset - First 3 batches]")
    for i, batch in enumerate(train_dataset):
        if i >= 3:
            break
        print(f"  Batch {i}:")
        print(
            f"    input_ids shape: {batch['input_ids'].shape}, dtype: {batch['input_ids'].dtype}"
        )
        print(
            f"    labels shape: {batch['labels'].shape}, dtype: {batch['labels'].dtype}"
        )
        print(f"    puzzle_identifiers shape: {batch['puzzle_identifiers'].shape}")
        print(f"    input_ids sample: {batch['input_ids'][0, :10].tolist()}")
        print(f"    labels sample: {batch['labels'][0, :10].tolist()}")

    # Test test set iteration
    print("\n[Test Dataset - First 3 batches]")
    for i, batch in enumerate(test_dataset):
        if i >= 3:
            break
        print(f"  Batch {i}:")
        print(f"    input_ids shape: {batch['input_ids'].shape}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
