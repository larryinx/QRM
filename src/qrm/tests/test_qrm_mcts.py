"""Smoke tests for QRM MCTS (requires GPU)."""

import torch

from qrm.losses import IGNORE_LABEL_ID
from qrm.models.qrm.configuration_qrm import QRMConfig
from qrm.models.qrm.modeling_qrm import QRMForPuzzleSolving


def _make_batch(batch_size: int, seq_len: int, vocab_size: int, device) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[:, :2] = IGNORE_LABEL_ID
    return {"input_ids": input_ids, "labels": labels}


def _tiny_qrm_config(**kwargs) -> QRMConfig:
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


def test_forward_creates_tree_and_returns_loss():
    """forward() initializes MCTS tree and returns loss."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    config = _tiny_qrm_config()

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, M, N = 2, config.fsq_top_k, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    output = model(**batch)

    assert output.loss is not None and output.loss.requires_grad
    assert output.logits.shape == (B, M, N, config.vocab_size)
    assert output.weights.shape == (B, M)
    assert model._mcts is not None


def test_puct_priors():
    """forward() returns priors when search_rule=puct."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    config = _tiny_qrm_config(search_rule="puct")

    with torch.device(device):
        model = QRMForPuzzleSolving(config)
    model.train()

    B, M, N = 2, config.fsq_top_k, config.seq_len
    batch = _make_batch(B, N, config.vocab_size, device)

    output = model(**batch)

    assert output.priors is not None
    assert output.priors.shape == (B, M)


def test_multi_iteration_completes():
    """forward() called num_iterations times triggers all_finish."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    config = _tiny_qrm_config(num_iterations=4)

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

    assert output.all_finish.item()
    assert model._mcts is None
