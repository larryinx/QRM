"""ARC Evaluator for ARC-AGI benchmarks.

Implements pass@K evaluation with aggregated voting across checkpoints.
"""

import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numba import njit

# Dihedral transform inverse mapping: index -> inverse transform id
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetries (rotate, flip, mirror)."""
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)  # horizontal flip
    elif tid == 5:
        return np.flipud(arr)  # vertical flip
    elif tid == 6:
        return arr.T  # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse dihedral transform."""
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


ARCMaxGridSize = 30
PuzzleIdSeparator = "|||"


def arc_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    """Convert ARC grid (list of lists) to numpy array."""
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def grid_hash(grid: np.ndarray) -> str:
    """Compute SHA256 hash of a grid for content-based deduplication."""
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())

    return hashlib.sha256(b"".join(buffer)).hexdigest()


def inverse_aug(name: str) -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    """Parse augmentation info from puzzle name and return inverse transform function.

    Supports multiple identifier formats:
    - Full: base_id|||t{trans}|||color_perm (e.g., "puzzle|||t3|||0612397845")
    - No color: base_id|||t{trans} (e.g., "puzzle|||t3")
    - No aug / shared: base_id (e.g., "puzzle" or "__shared_puzzle_embedding__")
    """
    if PuzzleIdSeparator not in name:
        # No augmentation info, return identity
        return name, lambda x: x

    parts = name.split(PuzzleIdSeparator)

    if len(parts) == 3:
        # Full format: base_id|||t{trans}|||color_perm
        base_name, trans_str, perm = parts
        trans_id = int(trans_str[1:])  # Remove "t" letter
        inv_perm = np.argsort([int(c) for c in perm]).astype(np.uint8)

        def _map_grid_full(grid: np.ndarray) -> np.ndarray:
            return inv_perm[inverse_dihedral_transform(grid, trans_id)]

        return base_name, _map_grid_full

    elif len(parts) == 2:
        # No color format: base_id|||t{trans}
        base_name, trans_str = parts
        trans_id = int(trans_str[1:])  # Remove "t" letter

        def _map_grid_no_color(grid: np.ndarray) -> np.ndarray:
            return inverse_dihedral_transform(grid, trans_id)

        return base_name, _map_grid_no_color

    else:
        # Unexpected format, return identity
        return name.split(PuzzleIdSeparator)[0], lambda x: x


@njit
def _crop(grid: np.ndarray) -> np.ndarray:
    """Find maximum-sized rectangle without any EOS token inside."""
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape

    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[: max_size[0], : max_size[1]] - 2).astype(np.uint8)


class ARC:
    """ARC Evaluator with pass@K metrics and aggregated voting support.

    Usage:
        1. Create evaluator instance
        2. Call begin_eval() to start evaluation
        3. Call update_batch(batch, preds) for each batch
        4. Call result() to get pass@K metrics

    Aggregated Voting:
        - aggregated_voting=True (default): Accumulate predictions across evaluation rounds
        - aggregated_voting=False: Clear predictions on each begin_eval()

    Identifier Modes:
        Supports multiple identifier modes via puzzle_aug_strings:
        - If batch contains "puzzle_aug_strings", uses that for inverse_aug
        - Otherwise falls back to identifier_map lookup (backward compatible)
    """

    required_outputs = {
        "inputs",
        "input_ids",
        "puzzle_identifiers",
        "puzzle_aug_strings",  # Optional: full augmentation strings for inverse_aug
        "q_halt_logits",
        "preds",
    }

    def __init__(
        self,
        data_path: str,
        blank_identifier_id: int,
        submission_K: int = 2,
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
        aggregated_voting: bool = True,
    ):
        """Initialize ARC Evaluator.

        Args:
            data_path: Data directory containing identifiers.json and test_puzzles.json.
            blank_identifier_id: Blank puzzle identifier ID for filtering padding.
            submission_K: Number of top-K predictions for submission.
            pass_Ks: List of K values for pass@K metrics.
            aggregated_voting: Whether to accumulate predictions across evaluations.
        """
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = blank_identifier_id

        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)

        self._local_hmap = {}
        self._local_preds = {}

    def begin_eval(self):
        """Begin evaluation. Clears state if aggregated_voting is disabled."""
        if not self.aggregated_voting:
            self._local_hmap = {}
            self._local_preds = {}

    def update_batch(
        self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]
    ):
        """Update evaluator state with predictions from a single batch.

        Args:
            batch: Dict containing "inputs" (or "input_ids"), "puzzle_identifiers",
                   and optionally "puzzle_aug_strings" for inverse_aug.
            preds: Dict containing "preds" and "q_halt_logits".
        """
        outputs = {}
        q_values = None
        aug_strings = None  # Optional: full augmentation strings

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    elif k == "puzzle_aug_strings":
                        # String arrays are numpy arrays, not tensors
                        aug_strings = v if isinstance(v, np.ndarray) else v
                    else:
                        outputs[k] = v.cpu() if hasattr(v, "cpu") else v

        assert q_values is not None, "q_halt_logits is required but not found"

        # Support both "input_ids" (QRM) and "inputs" (TRM) field names
        if "input_ids" in outputs and "inputs" not in outputs:
            outputs["inputs"] = outputs.pop("input_ids")

        # Remove padding
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        # SHARED mode: all data has puzzle_identifier=0, same as blank_identifier_id=0
        # Skip filtering to avoid removing all data (padding won't affect results
        # because its input_hash won't match any real test puzzle)
        if not mask.any():
            mask = torch.ones_like(mask, dtype=torch.bool)
        outputs = {k: v[mask] for k, v in outputs.items()}
        q_values = (
            q_values[mask.cpu()]
            if mask.device != torch.device("cpu")
            else q_values[mask]
        )
        if aug_strings is not None:
            aug_strings = aug_strings[mask.numpy() if hasattr(mask, "numpy") else mask]

        # Determine how to get augmentation strings
        use_aug_strings = aug_strings is not None

        for i, (identifier, input_grid, pred, q) in enumerate(
            zip(
                outputs["puzzle_identifiers"].numpy(),
                outputs["inputs"].numpy(),
                outputs["preds"].numpy(),
                q_values.numpy(),
            )
        ):
            # Get full augmentation string for inverse_aug
            if use_aug_strings:
                name = aug_strings[i]
            else:
                name = self.identifier_map[identifier]

            orig_name, _inverse_fn = inverse_aug(name)

            input_hash = grid_hash(_inverse_fn(_crop(input_grid)))

            pred = _inverse_fn(_crop(pred))
            assert np.all(
                (pred >= 0) & (pred <= 9)
            ), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred

            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))

    def result(self, save_path: Optional[str] = None) -> Dict[str, float]:
        """Compute final evaluation results.

        Args:
            save_path: Directory to save submission.json, None to skip saving.

        Returns:
            Dictionary of pass@K metrics.
        """
        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]

        for name, puzzle in self.test_puzzles.items():
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]

            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                # Aggregate votes by prediction hash
                p_map = {}
                for h, q in self._local_preds.get(name, {}).get(input_hash, []):
                    p_map.setdefault(h, [0, 0])
                    p_map[h][0] += 1  # count
                    p_map[h][1] += q  # sum of q values

                if not len(p_map):
                    print(f"Puzzle {name} has no predictions.")
                    continue

                # Compute average Q value
                for h, stats in p_map.items():
                    stats[1] /= stats[0]

                # Sort by [count, avg_q] descending
                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # Check pass@K for different K values
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                    num_test_correct[i] += ok

                # Get top predictions for submission
                pred_grids = []
                for h, stats in p_map[: self.submission_K]:
                    if h in self._local_hmap:
                        pred_grids.append(self._local_hmap[h])

                # Pad to submission_K
                while len(pred_grids) < self.submission_K:
                    if pred_grids:
                        pred_grids.append(pred_grids[0])
                    else:
                        pred_grids.append(np.zeros((1, 1), dtype=np.uint8))

                submission[name].append(
                    {
                        f"attempt_{i + 1}": grid.tolist()
                        for i, grid in enumerate(pred_grids)
                    }
                )

            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        all_results = {
            f"ARC/pass@{k}": correct[i] / len(self.test_puzzles)
            for i, k in enumerate(self.pass_Ks)
        }

        return all_results

    def state_dict(self) -> Dict:
        """Save evaluator state for cross-checkpoint aggregation.

        Returns:
            State dictionary containing hmap and preds.
        """
        return {
            "hmap": self._local_hmap.copy(),
            "preds": {
                puzzle_name: {
                    input_hash: list(pred_list)
                    for input_hash, pred_list in input_dict.items()
                }
                for puzzle_name, input_dict in self._local_preds.items()
            },
        }

    def load_state_dict(self, state: Dict):
        """Load evaluator state.

        Args:
            state: State dictionary from state_dict().
        """
        self._local_hmap = state["hmap"].copy()
        self._local_preds = {
            puzzle_name: {
                input_hash: list(pred_list)
                for input_hash, pred_list in input_dict.items()
            }
            for puzzle_name, input_dict in state["preds"].items()
        }

    def merge_state_dict(self, state: Dict):
        """Merge another evaluator's state (for multi-checkpoint aggregation).

        Args:
            state: State dictionary from state_dict().
        """
        self._local_hmap.update(state["hmap"])

        for puzzle_name, input_dict in state["preds"].items():
            self._local_preds.setdefault(puzzle_name, {})
            for input_hash, pred_list in input_dict.items():
                self._local_preds[puzzle_name].setdefault(input_hash, [])
                self._local_preds[puzzle_name][input_hash].extend(pred_list)
