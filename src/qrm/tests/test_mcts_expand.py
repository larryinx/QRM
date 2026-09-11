"""Model + MCTS integration tests for QRM (requires GPU)."""

import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

from qrm.losses import IGNORE_LABEL_ID
from qrm.models.qrm.configuration_qrm import QRMConfig
from qrm.models.qrm.modeling_qrm import QRMForPuzzleSolving, QRMInner, QRMInnerCarry


def _tiny_config(**kwargs):
    return QRMConfig(
        seq_len=8,
        vocab_size=11,
        hidden_size=64,
        num_heads=4,
        L_layers=1,
        H_cycles=2,
        L_cycles=2,
        halt_max_steps=3,
        fsq_levels=[8, 5, 5, 5],
        fsq_top_k=4,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=1,
        batch_size=2,
        forward_dtype="float32",
        **kwargs,
    )


def _make_batch(batch_size, seq_len, vocab_size, device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[:, :2] = IGNORE_LABEL_ID
    return {"input_ids": input_ids, "labels": labels}


def test_inner_forward_shapes():
    """QRMInner.forward returns correct shapes (6-tuple with priors)."""
    config = _tiny_config()
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, M, N = 2, config.fsq_top_k, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    carry = model.model.empty_carry(B, device=device)
    carry = model.model.reset_carry(torch.ones(B, dtype=torch.bool, device=device), carry)

    new_carry, logits, z_cont, z_quant, indices, priors = model.model(
        carry, batch, return_top_k=True,
    )

    assert new_carry.z_H.shape == (B, M, N, config.hidden_size), f"z_H: {new_carry.z_H.shape}"
    assert new_carry.z_L.shape == (B, M, N, config.hidden_size), f"z_L: {new_carry.z_L.shape}"
    assert logits.shape == (B, M, N, config.vocab_size), f"logits: {logits.shape}"
    assert z_cont.shape == (B, N, config.hidden_size), f"z_cont: {z_cont.shape}"
    assert z_quant.shape == (B, M, N, config.hidden_size), f"z_quant: {z_quant.shape}"
    assert indices.shape == (B, M, N), f"indices: {indices.shape}"
    assert priors is None, "priors should be None when need_priors=False"
    print("  PASS: inner_forward_shapes")


def test_inner_forward_with_priors():
    """QRMInner.forward returns priors when need_priors=True."""
    config = _tiny_config()
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, M, N = 2, config.fsq_top_k, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    carry = model.model.empty_carry(B, device=device)
    carry = model.model.reset_carry(torch.ones(B, dtype=torch.bool, device=device), carry)

    _, _, _, _, _, priors = model.model(
        carry, batch, return_top_k=True, need_priors=True,
    )

    assert priors is not None, "priors should not be None"
    assert priors.shape == (B, M), f"priors: {priors.shape}"
    prior_sums = priors.sum(dim=-1)
    assert torch.allclose(prior_sums, torch.ones_like(prior_sums), atol=1e-5), \
        f"priors should sum to 1, got {prior_sums}"
    print("  PASS: inner_forward_with_priors")


def test_single_forward_creates_tree():
    """One forward() call initializes MCTS tree and returns loss."""
    config = _tiny_config()
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, M, N = 2, config.fsq_top_k, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    output = model(**batch)

    assert output.loss is not None, "loss should not be None"
    assert output.loss.requires_grad, "loss should require grad"
    assert output.logits.shape == (B, M, N, config.vocab_size)
    assert output.weights.shape == (B, M)
    assert output.all_finish.item() == False, "Should not finish after 1 step"
    assert model._mcts is not None, "MCTS tree should be initialized"
    assert model._carry.steps.tolist() == [1, 1], "Steps should be 1"
    print("  PASS: single_forward_creates_tree")


def test_loss_backward():
    """Loss from forward() is differentiable."""
    config = _tiny_config()
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, N = 2, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    output = model(**batch)
    output.loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "At least one parameter should have a gradient"
    print("  PASS: loss_backward")


def test_multi_step_training():
    """Multiple forward calls build the tree and finish at num_iterations."""
    config = _tiny_config(num_iterations=5)
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, N = 2, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    for step in range(config.num_iterations):
        model.zero_grad()
        output = model(**batch)
        if output.loss.requires_grad:
            output.loss.backward()

        if step < config.num_iterations - 1:
            assert not output.all_finish.item(), f"Should not finish at step {step}"
        else:
            assert output.all_finish.item(), "Should finish at last step"

    # After all_finish, tree should be reset
    assert model._mcts is None, "MCTS tree should be reset after all_finish"
    assert (model._carry.steps == 0).all(), "Steps should reset after all_finish"
    print("  PASS: multi_step_training")


def test_select_best_carry():
    """select_best_carry reduces [B, M, S, D] to [B, S, D]."""
    device = torch.device("cuda:0")

    B, M, S, D = 2, 4, 8, 64
    carry = QRMInnerCarry(
        z_H=torch.randn(B, M, S, D, device=device),
        z_L=torch.randn(B, M, S, D, device=device),
    )
    best_indices = torch.tensor([2, 0], device=device)

    selected = QRMInner.select_best_carry(carry, best_indices)
    assert selected.z_H.shape == (B, S, D)
    assert selected.z_L.shape == (B, S, D)
    assert torch.equal(selected.z_H[0], carry.z_H[0, 2])
    assert torch.equal(selected.z_H[1], carry.z_H[1, 0])
    print("  PASS: select_best_carry")


def test_inference_multi_step():
    """Inference mode: forward for halt_max_steps, check all_finish."""
    config = _tiny_config()
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.eval()
    model.reset_carry()

    B, N = 2, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    with torch.no_grad():
        for step in range(config.halt_max_steps):
            output = model(**batch)

        assert output.all_finish.item(), "Should be finished after halt_max_steps"
        assert (model._carry.steps == 0).all(), "Steps should reset after all_finish"
    print("  PASS: inference_multi_step")


def test_second_sequence_after_reset():
    """After all_finish, a new sequence starts fresh tree."""
    config = _tiny_config(num_iterations=3)
    device = torch.device("cuda:0")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, N = 2, config.seq_len
    batch1 = _make_batch(B, N, config.vocab_size, device)

    # First sequence
    for _ in range(config.num_iterations):
        model.zero_grad()
        output = model(**batch1)
        if output.loss.requires_grad:
            output.loss.backward()

    assert model._mcts is None, "Tree should be None after finish"

    # Second sequence with different data
    batch2 = _make_batch(B, N, config.vocab_size, device)
    output2 = model(**batch2)
    assert output2.loss is not None
    assert model._mcts is not None, "New tree should be created"
    assert (model._carry.steps == 1).all(), "Steps should be 1 for new sequence"
    print("  PASS: second_sequence_after_reset")


def main():
    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
        return 0

    print("=== MCTS Integration Tests (GPU) ===\n")

    tests = [
        test_inner_forward_shapes,
        test_inner_forward_with_priors,
        test_single_forward_creates_tree,
        test_loss_backward,
        test_multi_step_training,
        test_select_best_carry,
        test_inference_multi_step,
        test_second_sequence_after_reset,
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
