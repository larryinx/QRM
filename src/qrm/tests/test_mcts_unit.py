"""Unit tests for QRM MCTS logic (no GPU needed)."""

import math
import sys
import torch

from qrm.mcts.node import QRMNode
from qrm.mcts.scoring import QRMSearchScorer
from qrm.mcts.search import QRMMCTSManager


class FakeCarry:
    """Minimal carry stub for testing MCTS logic without a real model."""
    def __init__(self, z_H=None, z_L=None, batch_size=1, dim=4):
        self.z_H = z_H if z_H is not None else torch.zeros(batch_size, 2, dim)
        self.z_L = z_L if z_L is not None else torch.zeros(batch_size, 2, dim)


def test_tree_construction():
    """Root creation and basic properties."""
    carry = FakeCarry()
    mgr = QRMMCTSManager(max_depth=3, search_rule="uct", c_uct=1.0, c_puct=1.0)
    mgr.initialize_roots(carry)

    assert len(mgr.roots) == 1
    root = mgr.roots[0]
    assert root.depth == 0
    assert root.is_leaf
    assert root.visit_count == 0
    assert root.q_value == 0.0
    print("  PASS: tree_construction")


def test_uct_scoring():
    """UCT formula produces expected values."""
    scorer = QRMSearchScorer("uct", c_uct=1.0, c_puct=0.0)

    parent = QRMNode(carry=None, depth=0, batch_index=0)
    parent.visit_count = 10
    parent.reward_sum = 5.0

    child_a = QRMNode(carry=None, depth=1, batch_index=0, parent=parent)
    child_a.visit_count = 3
    child_a.reward_sum = 2.4
    child_a.immediate_reward = 0.8

    child_b = QRMNode(carry=None, depth=1, batch_index=0, parent=parent)
    child_b.visit_count = 1
    child_b.reward_sum = 0.5
    child_b.immediate_reward = 0.5

    score_a = scorer.score_child(parent, child_a)
    score_b = scorer.score_child(parent, child_b)

    # Verify exploitation + exploration
    expected_a = 2.4 / 3.0 + 1.0 * math.sqrt(math.log(11.0) / 4.0)
    expected_b = 0.5 / 1.0 + 1.0 * math.sqrt(math.log(11.0) / 2.0)
    assert abs(score_a - expected_a) < 1e-6, f"{score_a} != {expected_a}"
    assert abs(score_b - expected_b) < 1e-6, f"{score_b} != {expected_b}"
    print("  PASS: uct_scoring")


def test_puct_scoring():
    """PUCT formula uses priors correctly."""
    scorer = QRMSearchScorer("puct", c_uct=0.0, c_puct=2.0)

    parent = QRMNode(carry=None, depth=0, batch_index=0)
    parent.visit_count = 5

    child = QRMNode(carry=None, depth=1, batch_index=0, parent=parent)
    child.visit_count = 2
    child.reward_sum = 1.0
    child.prior = 0.7
    child.immediate_reward = 0.5

    score = scorer.score_child(parent, child)
    expected = 0.5 + 2.0 * 0.7 * math.sqrt(6.0) / 3.0
    assert abs(score - expected) < 1e-6, f"{score} != {expected}"
    print("  PASS: puct_scoring")


def test_selection_picks_best():
    """select_child returns highest-scoring child."""
    scorer = QRMSearchScorer("uct", c_uct=0.0, c_puct=0.0)  # c=0 -> pure exploitation

    parent = QRMNode(carry=None, depth=0, batch_index=0)
    parent.visit_count = 10

    children = []
    for i, reward in enumerate([0.3, 0.9, 0.6]):
        child = QRMNode(carry=None, depth=1, batch_index=0, parent=parent)
        child.visit_count = 1
        child.reward_sum = reward
        child.immediate_reward = reward
        children.append(child)
    parent.children = children

    best = scorer.select_child(parent, children)
    assert best is children[1], "Should pick child with highest q_value"
    print("  PASS: selection_picks_best")


def test_backpropagation():
    """Backprop updates visit counts and reward sums up the tree."""
    scorer = QRMSearchScorer("uct", c_uct=1.0, c_puct=0.0)

    root = QRMNode(carry=None, depth=0, batch_index=0)
    child = QRMNode(carry=None, depth=1, batch_index=0, parent=root)
    grandchild = QRMNode(carry=None, depth=2, batch_index=0, parent=child)
    grandchild.immediate_reward = 0.8

    scorer.backpropagate(grandchild, grandchild.immediate_reward)

    assert grandchild.visit_count == 1
    assert grandchild.reward_sum == 0.8
    assert child.visit_count == 1
    assert child.reward_sum == 0.8
    assert root.visit_count == 1
    assert root.reward_sum == 0.8
    print("  PASS: backpropagation")


def test_terminal_skip_saturated():
    """skip_saturated: boundary parent becomes unselectable after first expansion."""
    carry = FakeCarry()
    mgr = QRMMCTSManager(
        max_depth=2, search_rule="uct", c_uct=1.0, c_puct=0.0,
        terminal_selection_mode="skip_saturated",
    )
    mgr.initialize_roots(carry)

    # First selection: root at depth 0
    sel = mgr.select_node(0)
    assert sel is not None
    assert sel.node.depth == 0

    # Expand root -> children at depth 1 (= max_depth - 1)
    child_carry = FakeCarry()
    child_carry.z_H = child_carry.z_H.unsqueeze(1).expand(-1, 2, -1, -1)  # [1, 2, S, D]
    child_carry.z_L = child_carry.z_L.unsqueeze(1).expand(-1, 2, -1, -1)
    mgr.sync_children_from_expansion(
        sel.node, child_carry,
        rewards=torch.tensor([0.5, 0.8]),
        priors=None,
        token_accuracy=torch.tensor([0.5, 0.8]),
        exact_match=torch.tensor([0.0, 0.0]),
    )
    mgr.backpropagate(sel.node.children[1])

    # Second selection: children at depth 1 = max_depth-1
    # After expansion, skip_saturated marks the parent (root) as terminal_closed
    # But root itself wasn't marked. The children at depth 1 can still be selected once.
    sel2 = mgr.select_node(0)
    assert sel2 is not None
    assert sel2.node.depth == 1  # Selects a child at boundary depth

    # Expand child at depth 1 -> grandchildren at depth 2 = max_depth
    child_carry2 = FakeCarry()
    child_carry2.z_H = child_carry2.z_H.unsqueeze(1).expand(-1, 2, -1, -1)
    child_carry2.z_L = child_carry2.z_L.unsqueeze(1).expand(-1, 2, -1, -1)
    mgr.sync_children_from_expansion(
        sel2.node, child_carry2,
        rewards=torch.tensor([0.6, 0.7]),
        priors=None,
        token_accuracy=torch.tensor([0.6, 0.7]),
        exact_match=torch.tensor([0.0, 0.0]),
    )
    mgr.backpropagate(sel2.node.children[1])

    # After both children at depth 1 are expanded, tree is fully saturated
    # The other child at depth 1 hasn't been expanded yet, so it's still selectable
    sel3 = mgr.select_node(0)
    # sel3 should select the unexpanded child at depth 1
    if sel3 is not None:
        assert sel3.node.depth == 1
        # Expand it too
        mgr.sync_children_from_expansion(
            sel3.node, child_carry2,
            rewards=torch.tensor([0.3, 0.4]),
            priors=None,
            token_accuracy=torch.tensor([0.3, 0.4]),
            exact_match=torch.tensor([0.0, 0.0]),
        )
        mgr.backpropagate(sel3.node.children[1])

    # Now all boundary parents are expanded, tree should be fully saturated
    sel4 = mgr.select_node(0)
    assert sel4 is None, "All nodes should be saturated"
    print("  PASS: terminal_skip_saturated")


def test_terminal_replay_parent():
    """replay_parent: boundary parent can be re-expanded."""
    carry = FakeCarry()
    mgr = QRMMCTSManager(
        max_depth=2, search_rule="uct", c_uct=1.0, c_puct=0.0,
        terminal_selection_mode="replay_parent",
    )
    mgr.initialize_roots(carry)

    # First expansion
    sel = mgr.select_node(0)
    child_carry = FakeCarry()
    child_carry.z_H = child_carry.z_H.unsqueeze(1).expand(-1, 2, -1, -1)
    child_carry.z_L = child_carry.z_L.unsqueeze(1).expand(-1, 2, -1, -1)
    mgr.sync_children_from_expansion(
        sel.node, child_carry,
        rewards=torch.tensor([0.5, 0.8]),
        priors=None,
        token_accuracy=torch.tensor([0.5, 0.8]),
        exact_match=torch.tensor([0.0, 0.0]),
    )
    mgr.backpropagate(sel.node.children[1])

    # Should keep being selectable (replay mode)
    for _ in range(5):
        sel = mgr.select_node(0)
        assert sel is not None, "replay_parent should always have selectable work"
    print("  PASS: terminal_replay_parent")


def test_get_best_leaf():
    """get_best_leaf returns leaf with highest q_value."""
    carry = FakeCarry()
    mgr = QRMMCTSManager(max_depth=3, search_rule="uct", c_uct=1.0, c_puct=0.0)
    mgr.initialize_roots(carry)

    root = mgr.roots[0]
    # Manually build tree
    for i in range(3):
        child = QRMNode(
            carry=FakeCarry(), depth=1, batch_index=0, parent=root,
            immediate_reward=0.1 * (i + 1),
        )
        child.visit_count = 1
        child.reward_sum = child.immediate_reward
        root.children.append(child)

    best = mgr.get_best_leaf(0)
    assert abs(best.immediate_reward - 0.3) < 1e-9, f"Expected ~0.3, got {best.immediate_reward}"
    print("  PASS: get_best_leaf")


def main():
    print("=== MCTS Unit Tests ===\n")

    tests = [
        test_tree_construction,
        test_uct_scoring,
        test_puct_scoring,
        test_selection_picks_best,
        test_backpropagation,
        test_terminal_skip_saturated,
        test_terminal_replay_parent,
        test_get_best_leaf,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
