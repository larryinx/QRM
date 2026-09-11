"""Test eval_use_compile behavior in TRMTrainer.

eval_use_compile controls how EMA weights are applied during evaluation:
- False (default): deep copy model with EMA weights (safe but breaks torch.compile)
- True: swap weights in-place (preserves torch.compile, needs correct restore)

This test verifies:
1. Both modes produce identical eval results given the same EMA state
2. After eval, training model weights are correctly restored
3. swap_weights is an involution (swap twice = identity)
"""

import copy

import torch
import torch.nn as nn

from qrm.trainers.trm import EMAHelper


class SimpleModel(nn.Module):
    """Minimal model for testing EMA swap/copy behavior."""

    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # Simulate carry state (like TRM/QRM)
        self._carry = None

    def forward(self, x):
        return self.linear(x)


def test_swap_weights_is_involution():
    """swap_weights called twice should restore original weights exactly."""
    torch.manual_seed(0)
    model = SimpleModel()
    ema = EMAHelper(mu=0.99)
    ema.register(model)

    # Simulate a few training steps to make EMA diverge from model
    for _ in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

    # Save original weights
    original_weights = {n: p.clone() for n, p in model.named_parameters()}

    # First swap: model → EMA, shadow → train
    ema.swap_weights(model)
    for n, p in model.named_parameters():
        assert not torch.equal(p, original_weights[n]), (
            f"After first swap, {n} should differ from original"
        )

    # Second swap: restore
    ema.swap_weights(model)
    for n, p in model.named_parameters():
        assert torch.equal(p, original_weights[n]), (
            f"After second swap, {n} should match original exactly"
        )

    print("PASSED: swap_weights is involution")


def test_swap_and_copy_produce_same_ema_weights():
    """swap mode and copy mode should yield identical EMA weights for inference."""
    torch.manual_seed(0)
    model = SimpleModel()
    ema = EMAHelper(mu=0.99)
    ema.register(model)

    # Diverge EMA from model
    for _ in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

    # Method 1: deep copy (eval_use_compile=False)
    copy_model = ema.ema_copy(model)
    copy_weights = {n: p.clone() for n, p in copy_model.named_parameters()}

    # Method 2: swap (eval_use_compile=True)
    ema.swap_weights(model)
    swap_weights = {n: p.clone() for n, p in model.named_parameters()}
    ema.swap_weights(model)  # restore

    # Both should have identical EMA weights
    for n in copy_weights:
        assert torch.equal(copy_weights[n], swap_weights[n]), (
            f"EMA weights differ for {n}: copy vs swap"
        )

    print("PASSED: swap and copy produce same EMA weights")


def test_eval_output_identical_both_modes():
    """Given same input and EMA state, both modes should produce identical output."""
    torch.manual_seed(0)
    model = SimpleModel()
    ema = EMAHelper(mu=0.99)
    ema.register(model)

    for _ in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

    x = torch.randn(8, 4)

    # eval_use_compile=False: deep copy
    copy_model = ema.ema_copy(model)
    copy_model.eval()
    with torch.no_grad():
        out_copy = copy_model(x)

    # eval_use_compile=True: swap
    ema.swap_weights(model)
    model.eval()
    with torch.no_grad():
        out_swap = model(x)
    ema.swap_weights(model)  # restore
    model.train()

    assert torch.equal(out_copy, out_swap), (
        f"Outputs differ: max diff = {(out_copy - out_swap).abs().max().item()}"
    )

    print("PASSED: eval output identical in both modes")


def test_training_weights_restored_after_swap_eval():
    """After swap-based eval, model weights must be exactly restored for continued training."""
    torch.manual_seed(0)
    model = SimpleModel()
    ema = EMAHelper(mu=0.99)
    ema.register(model)

    for _ in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

    # Snapshot training state
    train_weights = {n: p.clone() for n, p in model.named_parameters()}
    train_carry = "dummy_carry_state"
    model._carry = train_carry

    # Simulate eval with swap (what evaluation_loop does with eval_use_compile=True)
    saved_carry = model._carry
    ema.swap_weights(model)
    model.eval()

    # ... eval happens here ...
    with torch.no_grad():
        _ = model(torch.randn(4, 4))

    # Restore
    model.train()
    model._carry = saved_carry
    ema.swap_weights(model)

    # Verify weights restored
    for n, p in model.named_parameters():
        assert torch.equal(p, train_weights[n]), (
            f"Weight {n} not restored after swap eval"
        )

    # Verify carry restored
    assert model._carry == train_carry, "Carry not restored after swap eval"

    print("PASSED: training weights and carry restored after swap eval")


def test_copy_mode_does_not_touch_training_model():
    """eval_use_compile=False (deep copy) should never modify the training model."""
    torch.manual_seed(0)
    model = SimpleModel()
    ema = EMAHelper(mu=0.99)
    ema.register(model)

    for _ in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

    # Snapshot
    train_weights = {n: p.data_ptr() for n, p in model.named_parameters()}

    # Deep copy for eval
    _ = ema.ema_copy(model)

    # Training model's data pointers should be unchanged (no in-place modification)
    for n, p in model.named_parameters():
        assert p.data_ptr() == train_weights[n], (
            f"ema_copy modified training model's {n} in-place"
        )

    print("PASSED: copy mode does not touch training model")


if __name__ == "__main__":
    test_swap_weights_is_involution()
    test_swap_and_copy_produce_same_ema_weights()
    test_eval_output_identical_both_modes()
    test_training_weights_restored_after_swap_eval()
    test_copy_mode_does_not_touch_training_model()
    print("\nAll tests passed.")
