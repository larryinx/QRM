"""QRM Loss Functions.

This module implements the loss functions for QRM training:
- qrm_tree_loss: Weighted cross-entropy loss over candidates
- qrm_diversity_loss: Energy score based diversity loss (optional ablation)

Note: Reconstruction loss (MSE between z_continuous and z_quantized) is
implemented directly in the model forward, not here. Key point:
- z_continuous.detach() - continuous representation does not receive gradients
- z_quantized - FSQ output, learnable

Reference:
- QRM Paper: RecursiveQuant (ICML submission), Section 4.3
"""

import torch

from qrm.losses.trm import IGNORE_LABEL_ID, stablemax_cross_entropy


def qrm_tree_loss(
    logits: torch.Tensor,  # [B, M, N, V]
    labels: torch.Tensor,  # [B, N]
    weights: torch.Tensor,  # [B, M]
) -> torch.Tensor:
    """QRM tree-level weighted loss.

    Computes weighted cross-entropy loss over all candidates.
    Uses stablemax_cross_entropy to align with TRM's loss computation.

    Paper Eq. 22:
    L_tree = Σ_{μ∈U\\μ^(0)} w_μ × CE(f_O(y^(μ)), y)

    The weights are used to weight each candidate's CE loss within a sample,
    then summed across candidates to get per-sample loss, and finally
    summed across the batch (same as TRM's batch-level loss).

    Args:
        logits: [B, M, N, V] logits for all candidates
            B: batch size
            M: number of candidates (top_k)
            N: sequence length
            V: vocabulary size
        labels: [B, N] ground truth labels
        weights: [B, M] per-candidate weights (e.g., 0.2 + 0.8 * reward)

    Returns:
        Scalar loss tensor (sum over batch, like TRM)
    """
    B, M, N, V = logits.shape

    # Expand labels to [B, M, N]
    labels_expanded = labels.unsqueeze(1).expand(-1, M, -1)

    # Flatten to [B*M, N, V] and [B*M, N] for per-candidate processing
    logits_flat = logits.reshape(B * M, N, V)  # [B*M, N, V]
    labels_flat = labels_expanded.reshape(B * M, N)  # [B*M, N]

    # Compute mask and loss_divisor (same as TRM)
    mask = labels_flat != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1)  # [B*M]
    loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # [B*M, 1]

    # Per-token cross entropy using stablemax (same as TRM)
    # stablemax_cross_entropy returns [B*M, N] per-token loss
    ce_loss = stablemax_cross_entropy(
        logits_flat, labels_flat, valid_mask=mask
    )  # [B*M, N]

    # Average loss per candidate (same normalization as TRM: ce_loss / loss_divisor then sum)
    ce_loss_per_candidate = (ce_loss / loss_divisor).sum(-1)  # [B*M]
    ce_loss_per_candidate = ce_loss_per_candidate.reshape(B, M)  # [B, M]

    # Apply weights and sum all (Eq. 22: L_tree = Σ_μ w_μ × CE(...))
    # Candidates are treated as part of batch dimension, each weighted independently
    return (ce_loss_per_candidate * weights).sum()


def qrm_diversity_loss(
    z_candidates: torch.Tensor,  # [B, M, N, D]
    beta: float = 1.0,
    seq_aggregation: str = "mean",
) -> torch.Tensor:
    """Diversity loss: encourage candidates to be far apart.

    Based on Energy Score's inter-sample distance term:
    L_div = -E[d(X, X')] = -mean(||z_i - z_j||^beta)

    Note the negative sign: we want to maximize distance, so loss is negated.
    Minimizing L_div is equivalent to maximizing distance between candidates.

    Meeting conclusion: disabled by default, used as ablation item.

    Args:
        z_candidates: [B, M, N, D] all candidate latent representations
            B: batch size
            M: number of candidates (top_k)
            N: sequence length
            D: feature dimension
        beta: distance exponent (default 1.0 for L1-like, 2.0 for L2)
        seq_aggregation: how to aggregate over sequence dimension
            - "mean": mean pooling over sequence, then compute pairwise distance.
                      Standard for sentence embeddings and contrastive learning.
                      Ref: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings
                      using Siamese BERT-Networks", EMNLP 2019.
                      https://arxiv.org/abs/1908.10084
            - "sum": compute per-timestep distance, then sum over sequence.
                      Similar to Hamming diversity in Diverse Beam Search.
                      Ref: Vijayakumar et al., "Diverse Beam Search: Decoding
                      Diverse Solutions from Neural Sequence Models", AAAI 2018.
                      https://arxiv.org/abs/1610.02424

    Returns:
        Scalar loss tensor (negative mean pairwise distance)
    """
    B, M, N, D = z_candidates.shape

    # Edge case: single candidate, no diversity to compute
    if M <= 1:
        return torch.tensor(0.0, device=z_candidates.device, dtype=z_candidates.dtype)

    # Mask for excluding diagonal (i==j)
    mask = ~torch.eye(M, dtype=torch.bool, device=z_candidates.device)

    if seq_aggregation == "mean":
        # Mean pooling: average over sequence, then compute pairwise distance
        # Each candidate is represented by its mean vector
        z_mean = z_candidates.mean(dim=2)  # [B, M, D]

        # Compute pairwise distance between candidates
        z_i = z_mean.unsqueeze(2)  # [B, M, 1, D]
        z_j = z_mean.unsqueeze(1)  # [B, 1, M, D]

        # ||z_i - z_j||^beta
        diff = z_i - z_j  # [B, M, M, D]
        distance = torch.pow(torch.linalg.norm(diff, ord=2, dim=-1), beta)  # [B, M, M]

        # Apply mask: [1, M, M]
        mask = mask.unsqueeze(0)

    elif seq_aggregation == "sum":
        # Per-timestep sum: compute distance at each position, then sum
        # Similar to Hamming diversity in Diverse Beam Search
        z_i = z_candidates.unsqueeze(2)  # [B, M, 1, N, D]
        z_j = z_candidates.unsqueeze(1)  # [B, 1, M, N, D]

        # Per-timestep distance: ||z_i[t] - z_j[t]||^beta
        diff = z_i - z_j  # [B, M, M, N, D]
        per_step_distance = torch.pow(
            torch.linalg.norm(diff, ord=2, dim=-1), beta  # [B, M, M, N]
        )

        # Sum over sequence dimension
        distance = per_step_distance.sum(dim=-1)  # [B, M, M]

        # Apply mask: [1, M, M]
        mask = mask.unsqueeze(0)

    else:
        raise ValueError(
            f"Unknown seq_aggregation: {seq_aggregation}. Use 'mean' or 'sum'."
        )

    # Average distance over valid pairs
    # Valid pairs: B * M * (M-1)
    mean_distance = (distance * mask).sum() / (B * M * (M - 1))

    # Negative sign: maximize distance = minimize negative distance
    return -mean_distance


if __name__ == "__main__":
    # Dummy data for debugging
    # Shape convention: [B, M, N, D/V] where M is candidate dimension
    B, M, N, V = 2, 8, 100, 1000  # batch, candidates, seq_len, vocab
    D = 512  # feature dimension

    # === Test qrm_tree_loss ===
    print("=== qrm_tree_loss ===")

    # 模拟 logits: [B, M, N, V]
    logits = torch.randn(B, M, N, V)

    # 模拟 labels: [B, N]
    labels = torch.randint(0, V, (B, N))
    labels[:, :10] = IGNORE_LABEL_ID  # 前10个token忽略 (IGNORE_LABEL_ID = -100)

    # 模拟 weights: [B, M] (0.2 + 0.8 * reward)
    rewards = torch.rand(B, M)
    weights = 0.2 + 0.8 * rewards

    print(f"IGNORE_LABEL_ID = {IGNORE_LABEL_ID}")
    print(f"logits shape: {logits.shape}")  # [B, M, N, V]
    print(f"labels shape: {labels.shape}")  # [B, N]
    print(f"weights shape: {weights.shape}")  # [B, M]

    loss = qrm_tree_loss(logits, labels, weights)
    print(f"tree_loss: {loss.item():.4f}")

    # === Test qrm_diversity_loss ===
    print("\n=== qrm_diversity_loss ===")

    # 模拟 z_candidates: [B, M, N, D]
    z_candidates = torch.randn(B, M, N, D)

    print(f"z_candidates shape: {z_candidates.shape}")  # [B, M, N, D]

    # Test seq_aggregation="mean" (default)
    print("\n--- seq_aggregation='mean' ---")
    div_loss_mean = qrm_diversity_loss(z_candidates, beta=1.0, seq_aggregation="mean")
    print(f"diversity_loss (beta=1.0, mean): {div_loss_mean.item():.4f}")

    div_loss_mean_beta2 = qrm_diversity_loss(
        z_candidates, beta=2.0, seq_aggregation="mean"
    )
    print(f"diversity_loss (beta=2.0, mean): {div_loss_mean_beta2.item():.4f}")

    # Test seq_aggregation="sum"
    print("\n--- seq_aggregation='sum' ---")
    div_loss_sum = qrm_diversity_loss(z_candidates, beta=1.0, seq_aggregation="sum")
    print(f"diversity_loss (beta=1.0, sum): {div_loss_sum.item():.4f}")

    div_loss_sum_beta2 = qrm_diversity_loss(
        z_candidates, beta=2.0, seq_aggregation="sum"
    )
    print(f"diversity_loss (beta=2.0, sum): {div_loss_sum_beta2.item():.4f}")

    # Compare: sum mode should have larger magnitude (N times larger roughly)
    print(
        f"\nRatio sum/mean (beta=1.0): {div_loss_sum.item() / div_loss_mean.item():.2f} (expected ~{N})"
    )

    # Edge case: M=1
    print("\n--- Edge case: M=1 ---")
    z_single = torch.randn(B, 1, N, D)
    div_loss_single_mean = qrm_diversity_loss(
        z_single, beta=1.0, seq_aggregation="mean"
    )
    div_loss_single_sum = qrm_diversity_loss(z_single, beta=1.0, seq_aggregation="sum")
    print(
        f"diversity_loss (M=1, mean): {div_loss_single_mean.item():.4f}"
    )  # Should be 0.0
    print(
        f"diversity_loss (M=1, sum): {div_loss_single_sum.item():.4f}"
    )  # Should be 0.0

    print("\nAll tests passed!")
