"""StableMax loss functions."""

import torch

IGNORE_LABEL_ID = -100


def s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """StableMax transform function."""
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-StableMax."""
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """StableMax Cross Entropy."""
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)
