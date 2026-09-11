"""Stochastic Finite Scalar Quantization (FSQ) Module.

This module implements Stochastic FSQ with Top-K sampling for QRM.

Key features:
1. Joint probability-based quantization over codebook entries
2. Support for returning top-k candidates (for training/MCTS)
3. Support for do_sampling (for stochastic inference)
4. Straight-Through Estimator (STE) for gradient flow

Reference:
- FSQ Paper: https://arxiv.org/abs/2309.15505
- QRM Paper: RecursiveQuant (ICML submission), Section 4.1-4.2
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator."""
    return x + (x.round() - x).detach()


class StochasticFSQ(nn.Module):
    """Stochastic Finite Scalar Quantization with Top-K sampling.

    This module quantizes continuous latent representations into discrete codes
    using a finite set of quantization levels per dimension. It supports:
    - Deterministic (greedy) quantization
    - Stochastic sampling from top-k candidates
    - Returning all top-k candidates for MCTS training

    Args:
        levels: Number of quantization levels per dimension (e.g., [8, 5, 5, 5])
        dim: Input/output feature dimension
        top_k: Number of top candidates to consider
        temperature: Temperature for probability computation (lower = sharper)
        projection_has_bias: Whether projection layers have bias

    Example:
        >>> fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)
        >>> x = torch.randn(2, 100, 512)
        >>> # Training: return top-k candidates
        >>> out, indices = fsq(x, return_top_k=True)
        >>> # out: [2, 8, 100, 512], indices: [2, 8, 100]
        >>> # Inference with sampling
        >>> out, indices = fsq(x, return_top_k=False, do_sampling=True)
        >>> # out: [2, 1, 100, 512], indices: [2, 1, 100]

    Output shape convention:
        [B, M, N, D] where M is the candidate dimension.
        This treats each candidate as a complete sequence prediction,
        enabling B*M independent predictions for loss computation.
    """

    def __init__(
        self,
        levels: list[int] | tuple[int, ...] = (8, 5, 5, 5),
        dim: int = 512,
        top_k: int = 8,
        temperature: float = 1.0,
        projection_has_bias: bool = True,
        force_quantization_f32: bool = True,
        allowed_dtypes: tuple = (torch.float32, torch.float64),
        pre_norm: bool = False,
    ):
        super().__init__()

        self.levels = levels
        self.dim = dim
        self.top_k = top_k
        self.temperature = temperature
        self.force_quantization_f32 = force_quantization_f32
        self.allowed_dtypes = allowed_dtypes
        self.pre_norm = pre_norm

        # Codebook parameters
        self.codebook_dim = len(levels)
        self.codebook_size = math.prod(levels)

        # Warn if codebook size is very large (memory intensive)
        if self.codebook_size > 100000:
            import warnings

            warnings.warn(
                f"Codebook size {self.codebook_size} is very large. "
                f"Consider using smaller levels (current: {levels}). "
                f"Recommended: codebook_size < 10000 for efficiency.",
                UserWarning,
            )

        # Register levels as buffer for device/dtype handling
        self.register_buffer(
            "_levels", torch.tensor(levels, dtype=torch.int32), persistent=False
        )

        # Projection layers (dim != codebook_dim)
        self.project_in = nn.Linear(dim, self.codebook_dim, bias=projection_has_bias)
        self.project_out = nn.Linear(self.codebook_dim, dim, bias=projection_has_bias)

        # Precompute implicit codebook and level indices
        self._precompute_codebook()

    def _precompute_codebook(self):
        """Precompute implicit codebook and index mappings.

        This follows the same convention as vector_quantize_pytorch.FSQ:
        - basis = cumprod([1] + levels[:-1])
        - index = sum(level_idx * basis)

        For levels=[8,5,5,5]:
        - basis = [1, 8, 40, 200]
        - codebook_size = 1000
        """
        # Compute basis following FSQ convention
        # For levels=[8,5,5,5]: basis=[1, 8, 40, 200]
        basis = torch.cumprod(
            torch.tensor([1] + self.levels[:-1], dtype=torch.long), dim=0
        )
        self.register_buffer("_basis", basis, persistent=False)

        # Generate implicit codebook using the same method as FSQ
        # indices_to_codes: index -> level_indices -> normalized codes
        indices = torch.arange(self.codebook_size)
        level_indices = self._indices_to_level_indices(indices)  # [codebook_size, D]
        implicit_codebook = self._level_indices_to_codes(
            level_indices
        )  # [codebook_size, D]

        self.register_buffer("implicit_codebook", implicit_codebook)

        # Precompute unique levels per dimension and index mapping
        self._precompute_level_indices()

    def _indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices to per-dimension level indices.

        Args:
            indices: [...] flat codebook indices

        Returns:
            level_indices: [..., codebook_dim] per-dimension indices
        """
        # Same as FSQ: (indices // basis) % levels
        indices = indices.unsqueeze(-1)  # [..., 1]
        level_indices = (indices // self._basis) % self._levels.long()
        return level_indices

    def _level_indices_to_codes(self, level_indices: torch.Tensor) -> torch.Tensor:
        """Convert level indices to normalized codes in [-1, 1].

        Args:
            level_indices: [..., codebook_dim] per-dimension indices (0 to L-1)

        Returns:
            codes: [..., codebook_dim] normalized codes
        """
        # Same as FSQ: (level_idx - half_width) / half_width
        half_width = (self._levels // 2).float()
        codes = (level_indices.float() - half_width) / half_width
        return codes

    def _precompute_level_indices(self):
        """Precompute per-dimension quantization levels and index mapping.

        This enables efficient computation of joint probability by mapping
        each codebook entry to its per-dimension level indices.
        """
        D = self.codebook_dim

        # Extract unique quantization levels for each dimension
        # q_levels[d] contains the unique values in dimension d
        q_levels_list = []
        for d in range(D):
            unique_levels = self.implicit_codebook[:, d].unique(sorted=True)
            q_levels_list.append(unique_levels)

        # Store as a list of tensors (different sizes per dimension)
        self.q_levels = q_levels_list

        # codebook_indices[i, d] = level index of codebook entry i in dimension d
        # This is equivalent to _indices_to_level_indices(torch.arange(codebook_size))
        indices = torch.arange(self.codebook_size, device=self.implicit_codebook.device)
        codebook_indices = self._indices_to_level_indices(indices)
        self.register_buffer("codebook_indices", codebook_indices)

    def bound_soft(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Soft bound: map z to [-1, 1] without rounding.

        This is similar to FSQ's bound() but without round_ste,
        keeping continuous values for probability computation.

        Args:
            z: Input tensor [..., codebook_dim]
            eps: Small epsilon for numerical stability

        Returns:
            Bounded tensor in [-1, 1] range
        """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return bounded_z / half_width  # Normalize to [-1, 1]

    def bound_hard(self, z: torch.Tensor) -> torch.Tensor:
        """Hard bound with rounding (deterministic quantization).

        Args:
            z: Input tensor [..., codebook_dim]

        Returns:
            Quantized tensor with values at discrete levels
        """
        bounded = self.bound_soft(z)
        # Round to nearest level
        half_width = self._levels // 2
        scaled = bounded * half_width
        rounded = round_ste(scaled)
        return rounded / half_width

    def compute_joint_log_probs(
        self,
        bounded_z: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute joint log probability for each codebook entry.

        For each position, computes P(code) = ∏_d P(q_d | z_d)
        where P(q_d | z_d) is based on squared distance with Gibbs distribution.

        Args:
            bounded_z: [..., codebook_dim] soft-bounded continuous values
            temperature: Temperature parameter (lower = sharper distribution)

        Returns:
            joint_log_probs: [..., codebook_size] log probabilities
        """
        if temperature is None:
            temperature = self.temperature

        # Get shape info
        *batch_dims, D = bounded_z.shape
        device = bounded_z.device
        dtype = bounded_z.dtype

        # Flatten batch dimensions for easier processing
        bounded_z_flat = bounded_z.reshape(-1, D)  # [N, D]
        N = bounded_z_flat.shape[0]

        # Compute per-dimension log probabilities
        log_probs_per_dim = []
        for d in range(D):
            u = bounded_z_flat[:, d]  # [N]
            qd = self.q_levels[d].to(device)  # [K_d]

            # Squared distance: d_k = (u - q_k)²
            d2 = (u.unsqueeze(-1) - qd) ** 2  # [N, K_d]

            # Gibbs distribution: logits = -d² / σ²
            logits = -d2 / (temperature**2 + 1e-10)

            # Numerically stable log_softmax
            log_probs = F.log_softmax(logits, dim=-1)  # [N, K_d]
            log_probs_per_dim.append(log_probs)

        # Compute joint log probability for each codebook entry
        # P(code) = ∏_d P(q_d | z_d) => log P = Σ_d log P(q_d | z_d)
        joint_log_probs = torch.zeros(N, self.codebook_size, device=device, dtype=dtype)

        for d in range(D):
            # codebook_indices[:, d] maps each code to its level index in dimension d
            indices = self.codebook_indices[:, d]  # [codebook_size]
            indices_expanded = indices.unsqueeze(0).expand(N, -1)  # [N, codebook_size]

            # Gather the log probability for each code's level in dimension d
            selected_log_probs = torch.gather(
                log_probs_per_dim[d], dim=-1, index=indices_expanded
            )  # [N, codebook_size]

            joint_log_probs = joint_log_probs + selected_log_probs

        # Reshape back to original batch dimensions
        joint_log_probs = joint_log_probs.reshape(*batch_dims, self.codebook_size)

        return joint_log_probs

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices to codes.

        Args:
            indices: [...] integer indices into codebook

        Returns:
            codes: [..., codebook_dim] quantized codes
        """
        return self.implicit_codebook[indices]

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert codes to codebook indices.

        Same as FSQ.codes_to_indices:
        1. Scale codes from [-1, 1] to [0, L-1] level indices
        2. Compute flat index using basis

        Args:
            codes: [..., codebook_dim] quantized codes (values in [-1, 1])

        Returns:
            indices: [...] integer indices
        """
        # Scale codes to level indices: code -> level_idx
        # code = (level_idx - half_width) / half_width
        # => level_idx = code * half_width + half_width
        half_width = (self._levels // 2).float()
        level_indices = (codes * half_width + half_width).round().long()

        # Compute flat index: sum(level_idx * basis)
        indices = (level_indices * self._basis).sum(dim=-1)
        return indices

    def forward(
        self,
        z: torch.Tensor,
        return_top_k: bool = True,
        do_sampling: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with stochastic quantization.

        Args:
            z: [B, N, D] input latent tensor
            return_top_k: If True, return all top-k candidates
                         If False, return single candidate
            do_sampling: When return_top_k=False, whether to sample from top-k
                        (True) or use greedy/top-1 (False)

        Returns:
            quantized: [B, M, N, dim] where M=top_k or M=1
            indices: [B, M, N] codebook indices
        """
        from contextlib import nullcontext
        from functools import partial

        from torch.amp import autocast

        B, N, _ = z.shape
        device = z.device
        orig_dtype = z.dtype

        # Project to codebook dimension
        # Cast to weight dtype for compatibility with mixed precision training
        weight_dtype = self.project_in.weight.dtype
        z_proj = self.project_in(z.to(weight_dtype))  # [B, N, codebook_dim]

        # Pre-tanh normalization: standardize across positions per codebook dimension
        # Ensures each dim has σ≈1 before tanh, preventing saturation/bimodal collapse
        if self.pre_norm:
            mean = z_proj.mean(dim=-2, keepdim=True)   # [B, 1, C]
            std = z_proj.std(dim=-2, keepdim=True)      # [B, 1, C]
            z_proj = (z_proj - mean) / (std + 1e-5)     # [B, N, C], ~N(0,1) per dim

        # Whether to force quantization step to be full precision
        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, device.type, enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            # Convert to float32 if needed for numerical stability
            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z_proj = z_proj.float()

            # Soft bound to [-1, 1]
            bounded_z = self.bound_soft(z_proj)  # [B, N, codebook_dim]

            # Compute joint log probabilities
            joint_log_probs = self.compute_joint_log_probs(
                bounded_z
            )  # [B, N, codebook_size]

            # Select top-k codes
            topk_log_probs, topk_code_indices = torch.topk(
                joint_log_probs, k=self.top_k, dim=-1, largest=True
            )  # [B, N, top_k]

            if return_top_k:
                # Return all top-k candidates
                M = self.top_k
                indices = topk_code_indices  # [B, N, M]
            else:
                # Return single candidate
                if do_sampling:
                    # Sample from top-k according to probabilities
                    probs = F.softmax(topk_log_probs, dim=-1)  # [B, N, top_k]
                    probs_flat = probs.reshape(B * N, self.top_k)
                    sampled_idx = torch.multinomial(
                        probs_flat, num_samples=1
                    )  # [B*N, 1]
                    sampled_idx = sampled_idx.reshape(B, N, 1)
                    indices = torch.gather(topk_code_indices, dim=-1, index=sampled_idx)
                else:
                    # Greedy: select top-1
                    indices = topk_code_indices[..., :1]  # [B, N, 1]

            # Get codes from indices
            # indices: [B, N, M] -> need to get codes and reshape to [B, M, N, codebook_dim]
            M = indices.shape[-1]
            indices_flat = indices.reshape(-1)  # [B*N*M]
            codes_flat = self.implicit_codebook[indices_flat]  # [B*N*M, codebook_dim]
            codes = codes_flat.reshape(
                B, N, M, self.codebook_dim
            )  # [B, N, M, codebook_dim]

            # Straight-Through Estimator (STE)
            # Gradient flows through bounded_z, but forward uses discrete codes
            bounded_z_expanded = bounded_z.unsqueeze(2)  # [B, N, 1, codebook_dim]
            codes_ste = bounded_z_expanded + (codes - bounded_z_expanded).detach()

            # Convert back to original dtype
            codes_ste = codes_ste.to(orig_dtype)

            # Permute to [B, M, N, codebook_dim] - M candidates, each is a complete sequence
            codes_ste = codes_ste.permute(0, 2, 1, 3)  # [B, M, N, codebook_dim]

        # Project back to original dimension
        # Cast to weight dtype for compatibility with mixed precision training
        codes_ste_flat = codes_ste.reshape(B * M * N, self.codebook_dim)
        out_flat = self.project_out(codes_ste_flat.to(weight_dtype))  # [B*M*N, dim]
        out = out_flat.reshape(B, M, N, self.dim).to(orig_dtype)  # [B, M, N, dim]

        # Extract diagnostics before permute (indices still [B, N, M])
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "topk_indices": topk_code_indices.detach(),   # [B, N, top_k]
                "topk_log_probs": topk_log_probs.detach(),    # [B, N, top_k]
                "selected_indices": indices.detach(),          # [B, N, M]
                "bounded_z": bounded_z.detach(),               # [B, N, codebook_dim]
                "joint_log_probs": joint_log_probs.detach(),   # [B, N, codebook_size]
            }

        # Permute indices to match output shape: [B, N, M] -> [B, M, N]
        indices = indices.permute(0, 2, 1)  # [B, M, N]

        if return_diagnostics:
            return out, indices, diagnostics
        return out, indices

    def extra_repr(self) -> str:
        return (
            f"levels={self.levels}, dim={self.dim}, "
            f"codebook_size={self.codebook_size}, top_k={self.top_k}, "
            f"temperature={self.temperature}, pre_norm={self.pre_norm}"
        )


if __name__ == "__main__":
    fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)
    x = torch.randn(2, 100, 512)

    # 训练时：返回 top-k candidates
    out, indices = fsq(x, return_top_k=True)
    print(f"Training mode: out={out.shape}, indices={indices.shape}")
    # out: [2, 8, 100, 512], indices: [2, 8, 100]
    assert out.shape == (2, 8, 100, 512), f"Expected [2, 8, 100, 512], got {out.shape}"
    assert indices.shape == (2, 8, 100), f"Expected [2, 8, 100], got {indices.shape}"

    # 推理时 - 贪婪
    out, indices = fsq(x, return_top_k=False, do_sampling=False)
    print(f"Greedy mode: out={out.shape}, indices={indices.shape}")
    # out: [2, 1, 100, 512], indices: [2, 1, 100]
    assert out.shape == (2, 1, 100, 512), f"Expected [2, 1, 100, 512], got {out.shape}"
    assert indices.shape == (2, 1, 100), f"Expected [2, 1, 100], got {indices.shape}"

    # 推理时 - 随机采样
    out, indices = fsq(x, return_top_k=False, do_sampling=True)
    print(f"Sampling mode: out={out.shape}, indices={indices.shape}")
    # out: [2, 1, 100, 512], indices: [2, 1, 100]
    assert out.shape == (2, 1, 100, 512), f"Expected [2, 1, 100, 512], got {out.shape}"
    assert indices.shape == (2, 1, 100), f"Expected [2, 1, 100], got {indices.shape}"

    print("\nAll shape tests passed!")
