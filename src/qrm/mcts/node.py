from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QRMNode:
    """Search node for one detached carry state."""

    carry: Any
    depth: int
    batch_index: int
    parent: Optional["QRMNode"] = None
    candidate_idx_from_parent: Optional[int] = None
    prior: float = 1.0
    immediate_reward: float = 0.0
    token_accuracy: float = 0.0
    exact_match: float = 0.0
    visit_count: int = 0
    reward_sum: float = 0.0
    expanded: bool = False
    terminal_closed: bool = False
    children: list["QRMNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def q_value(self) -> float:
        if self.visit_count > 0:
            return self.reward_sum / self.visit_count
        return self.immediate_reward
