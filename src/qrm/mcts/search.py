from dataclasses import dataclass
from typing import Any, Optional

import torch

from qrm.mcts.node import QRMNode
from qrm.mcts.scoring import QRMSearchScorer


@dataclass
class QRMSelection:
    path: list[QRMNode]
    node: QRMNode


class QRMMCTSManager:
    """Manage one search tree per batch element."""

    def __init__(
        self,
        max_depth: int,
        search_rule: str,
        c_uct: float,
        c_puct: float,
        terminal_selection_mode: str = "skip_saturated",
        q_normalize: bool = True,
    ):
        if terminal_selection_mode not in {"replay_parent", "skip_saturated"}:
            raise ValueError(
                f"Unknown terminal selection mode: {terminal_selection_mode}"
            )

        self.max_depth = max_depth
        self.terminal_selection_mode = terminal_selection_mode
        self.scorer = QRMSearchScorer(
            search_rule, c_uct=c_uct, c_puct=c_puct, q_normalize=q_normalize,
        )
        self.roots: list[QRMNode] = []

    def initialize_roots(self, carry: Any) -> None:
        batch_size = carry.z_H.shape[0]
        self.roots = []
        for batch_idx in range(batch_size):
            self.roots.append(
                QRMNode(
                    carry=self._slice_carry(carry, batch_idx),
                    depth=0,
                    batch_index=batch_idx,
                    prior=1.0,
                )
            )

    def release(self) -> None:
        """Break parent↔child reference cycles so the tree's CUDA carries
        can be freed immediately by refcount GC.

        Without this, every tree node survives as part of a reference cycle
        (child.parent -> parent, parent.children -> child) and waits for
        Python's mark-sweep collector. That collector is triggered by CPython
        object allocations, not by GPU allocations, so several GB of dead
        carries can sit in CUDA memory across many iterations before being
        freed — rapidly OOMing the device.
        """
        stack = list(self.roots)
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            node.children = []
            node.parent = None
            node.carry = None
        self.roots = []

    def _slice_carry(self, carry: Any, batch_idx: int) -> Any:
        carry_cls = type(carry)
        return carry_cls(
            z_H=carry.z_H[batch_idx : batch_idx + 1].detach().clone(),
            z_L=carry.z_L[batch_idx : batch_idx + 1].detach().clone(),
        )

    def _candidate_carry(self, carry: Any, candidate_idx: int) -> Any:
        carry_cls = type(carry)
        return carry_cls(
            z_H=carry.z_H[:, candidate_idx].detach().clone(),
            z_L=carry.z_L[:, candidate_idx].detach().clone(),
        )

    def _node_has_selectable_work(self, node: QRMNode) -> bool:
        if node.depth >= self.max_depth:
            return False

        if node.depth == self.max_depth - 1:
            return self.terminal_selection_mode == "replay_parent" or not node.expanded

        if node.is_leaf:
            return True

        return any(self._node_has_selectable_work(child) for child in node.children)

    def select_node(self, batch_idx: int) -> Optional[QRMSelection]:
        root = self.roots[batch_idx]
        if not self._node_has_selectable_work(root):
            return None

        path = [root]
        node = root

        while True:
            if node.depth == self.max_depth - 1:
                return QRMSelection(path=path, node=node)

            if node.is_leaf:
                return QRMSelection(path=path, node=node)

            eligible_children = [
                child
                for child in node.children
                if self._node_has_selectable_work(child)
            ]
            if not eligible_children:
                return None

            node = self.scorer.select_child(node, eligible_children)
            path.append(node)

    def sync_children_from_expansion(
        self,
        node: QRMNode,
        child_carry: Any,
        rewards: torch.Tensor,
        priors: Optional[torch.Tensor],
        token_accuracy: torch.Tensor,
        exact_match: torch.Tensor,
    ) -> list[QRMNode]:
        num_children = rewards.shape[0]
        if priors is None:
            priors = torch.full_like(rewards, 1.0 / max(num_children, 1))

        if not node.children:
            for child_idx in range(num_children):
                node.children.append(
                    QRMNode(
                        carry=self._candidate_carry(child_carry, child_idx),
                        depth=node.depth + 1,
                        batch_index=node.batch_index,
                        parent=node,
                        candidate_idx_from_parent=child_idx,
                    )
                )

        if len(node.children) != num_children:
            raise ValueError(
                f"Expansion child count mismatch: existing={len(node.children)}, new={num_children}"
            )

        for child_idx, child in enumerate(node.children):
            child.prior = float(priors[child_idx].item())
            child.immediate_reward = float(rewards[child_idx].item())
            child.token_accuracy = float(token_accuracy[child_idx].item())
            child.exact_match = float(exact_match[child_idx].item())
            if child.depth >= self.max_depth:
                child.terminal_closed = self.terminal_selection_mode == "skip_saturated"

        node.expanded = True
        if node.depth == self.max_depth - 1 and self.terminal_selection_mode == "skip_saturated":
            node.terminal_closed = True

        return node.children

    def backpropagate(self, node: QRMNode) -> None:
        """Single-child backprop: walk up from `node` with its immediate reward."""
        self.scorer.backpropagate(node, node.immediate_reward)

    def backpropagate_all_children(self, parent: QRMNode) -> None:
        """Propagate the rewards of *every* newly expanded child up the tree.

        After `sync_children_from_expansion` has attached M children to
        `parent`, each child is treated as a completed simulation:

        - Every child's own visit/reward is incremented once:
            N(child) += 1,  W(child) += R_imm(child)
        - Every ancestor (`parent` and up) absorbs all M rewards together:
            N(ancestor) += M,  W(ancestor) += Σ_m R_imm(child_m)

        This matches the "propagate all candidates" UCT variant: the
        expansion step observed M supervised rewards simultaneously, so the
        statistics must reflect M simulations rather than a single best one.
        """
        if not parent.children:
            return

        total_reward = 0.0
        num_children = 0
        for child in parent.children:
            child.visit_count += 1
            child.reward_sum += child.immediate_reward
            total_reward += child.immediate_reward
            num_children += 1

        current = parent
        while current is not None:
            current.visit_count += num_children
            current.reward_sum += total_reward
            current = current.parent

    def get_best_leaf(self, batch_idx: int) -> QRMNode:
        root = self.roots[batch_idx]
        stack = [root]
        leaves = []

        while stack:
            node = stack.pop()
            if node.is_leaf or node.depth >= self.max_depth:
                leaves.append(node)
            else:
                stack.extend(node.children)

        if not leaves:
            return root
        return max(leaves, key=lambda node: node.q_value)
