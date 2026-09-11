import torch
import torch.nn.functional as F
from torch import nn


class FSQAttention(nn.Module):
    """Learned-query attention over continuous + discrete FSQ candidates.

    For each position n, computes a weighted sum over (K+1) candidates:
        z'_{L,n} = alpha_n^(0) * z_{L,n} + sum_{k=1}^{K} alpha_n^(k) * z_{Q,n}^(k)

    where alpha = softmax(omega @ KV^T / sqrt(D)) and omega is a single
    learned query vector shared across all positions and batches.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Learned query vector: zero-init gives uniform 1/(K+1) attention at start
        self.omega = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, z: torch.Tensor, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, S, D] original continuous hidden state
            z_quantized: [B, K, S, D] top-k quantized candidates

        Returns:
            [B, S, D] attention-weighted output over z and z_quantized
        """
        B, S, D = z.shape
        K = z_quantized.shape[1]

        # Build KV: prepend continuous z as candidate 0
        # z: [B, S, D] -> [B, S, 1, D]
        # z_quantized: [B, K, S, D] -> [B, S, K, D]
        # kv: [B, S, K+1, D] -> [B*S, K+1, D]
        z_expanded = z.unsqueeze(2)                      # [B, S, 1, D]
        z_q_perm = z_quantized.permute(0, 2, 1, 3)      # [B, S, K, D]
        kv = torch.cat([z_expanded, z_q_perm], dim=2)    # [B, S, K+1, D]
        kv = kv.reshape(B * S, K + 1, D)                 # [B*S, K+1, D]

        # Query: shared learned vector, broadcast to all positions
        # Cast to input dtype for mixed precision (bfloat16) compatibility
        q = self.omega.to(z.dtype).expand(B * S, 1, D)   # [B*S, 1, D]

        # Attention: [B*S, 1, D] x [B*S, D, K+1] -> [B*S, 1, K+1] x [B*S, K+1, D]
        out = F.scaled_dot_product_attention(q, kv, kv)  # [B*S, 1, D]

        return out.reshape(B, S, D)
