import math
from typing import Iterable

from qrm.mcts.node import QRMNode


class QRMSearchScorer:
    """UCT / PUCT scorer for QRM search nodes.

    UCT is the primary selection rule. The formula follows classic UCB1:

        UCT(s, a) = Q(s, a) + c_uct * sqrt( ln N(s) / N(s, a) )

    with the convention that unvisited children (N(s, a) == 0) return +inf so
    they are always selected first. The default c_uct = sqrt(2) matches UCB1.

    PUCT is retained for ablation (AlphaZero-style prior-weighted exploration)
    but is not the current default in QRMConfig.

    When q_normalize is True, Q(s, a) is clipped into [0, 1] before it is used
    as the exploitation term. Rewards are already bounded in [0, 1] by
    _compute_reward, so this is a safety clip rather than a true rescaling.
    """

    def __init__(
        self,
        rule: str,
        c_uct: float,
        c_puct: float,
        q_normalize: bool = True,
    ):
        if rule not in {"uct", "puct"}:
            raise ValueError(f"Unknown search rule: {rule}")
        self.rule = rule
        self.c_uct = c_uct
        self.c_puct = c_puct
        self.q_normalize = q_normalize

    def _q(self, child: QRMNode) -> float:
        q = child.q_value
        if self.q_normalize:
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
        return q

    def score_child(self, parent: QRMNode, child: QRMNode) -> float:
        if self.rule == "uct":
            # Unvisited children are always selected first.
            if child.visit_count == 0:
                return float("inf")
            exploitation = self._q(child)
            n_parent = max(parent.visit_count, 1)
            exploration = self.c_uct * math.sqrt(
                math.log(n_parent) / child.visit_count
            )
            return exploitation + exploration

        # PUCT (kept for ablation).
        exploitation = self._q(child)
        exploration = (
            self.c_puct
            * child.prior
            * math.sqrt(parent.visit_count + 1.0)
            / (child.visit_count + 1.0)
        )
        return exploitation + exploration

    def select_child(self, parent: QRMNode, children: Iterable[QRMNode]) -> QRMNode:
        children = list(children)
        if not children:
            raise ValueError("select_child called with no eligible children")

        best_child = None
        best_score = float("-inf")
        for child in children:
            score = self.score_child(parent, child)
            if math.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            return max(children, key=lambda child: self._q(child))
        return best_child

    def backpropagate(self, node: QRMNode, reward: float) -> None:
        """Classic single-path backprop: walk up from `node`, each ancestor
        gets N += 1, W += reward. Retained for compatibility and tests.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.reward_sum += reward
            current = current.parent
