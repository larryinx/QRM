from contextlib import nullcontext
from functools import partial

import torch
from einops import rearrange
from torch import autocast
from vector_quantize_pytorch import FSQ
from vector_quantize_pytorch.finite_scalar_quantization import (
    maybe,
    pack_one,
    unpack_one,
)


class StochasticFSQTopK(FSQ):
    """
    Stochastic Finite Scalar Quantization with Top-K sampling.

    Extends FSQ with probabilistic quantization based on joint probability over codebook entries.
    Instead of deterministic rounding, computes per-dimension probabilities and selects top-k
    codes with the highest joint probability.

    Args:
        levels: Number of quantization levels per dimension (e.g., [8,5,5,5])
        dim: Input feature dimension
        num_codebooks: Number of codebooks (only 1 supported)
        top_k: Number of top candidates to select/sample from
        temperature: Temperature for probability computation (lower = sharper)
        use_gumbel: Whether to use Gumbel noise (not implemented)
        do_sample: If False, falls back to deterministic FSQ
        return_top_k: If True, return all top-k codes; if False, sample one from top-k
    """

    def __init__(
        self,
        levels: list[int] | tuple[int, ...] = (8, 5, 5, 5),
        dim: int = 512,
        num_codebooks: int = 1,
        top_k: int = 8,
        temperature: float = 0.1,
        use_gumbel: bool = False,
        do_sample: bool = True,
        return_top_k: bool = True,
        **kwargs,
    ):
        if num_codebooks > 1:
            raise NotImplementedError
        if use_gumbel:
            raise NotImplementedError

        super(StochasticFSQTopK, self).__init__(dim=dim, levels=levels, **kwargs)

        self.top_k = top_k
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.do_sample = do_sample
        self.return_top_k = return_top_k

        # Extract unique quantization levels for each dimension
        self.q_levels = [
            self.implicit_codebook[:, lvl_idx].unique(sorted=True)
            for lvl_idx in range(len(levels))
        ]
        self.codebook_size = self.implicit_codebook.shape[0]

        # Precompute indices mapping: codebook_indices[i, d] = index of implicit_codebook[i, d] in q_levels[d]
        # This enables efficient lookup of per-dimension probabilities during joint probability computation
        self.register_buffer(
            "codebook_indices",
            torch.zeros(self.implicit_codebook.shape, dtype=torch.long),
        )
        for d in range(len(self.q_levels)):
            for idx in range(self.codebook_size):
                code_val = self.implicit_codebook[idx, d]
                self.codebook_indices[idx, d] = (self.q_levels[d] == code_val).nonzero(
                    as_tuple=True
                )[0][0]

    def bound_soft(self, z, eps=1e-3):
        """Soft bound: same as parent's bound() but without round_ste, keeps continuous values in [-1, 1]."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return (
            bounded_z / half_width
        )  # Normalize to [-1, 1] to match implicit_codebook range

    def symmetry_preserving_bound_soft(self, z):
        """Soft bound: same as parent's symmetry_preserving_bound() but without floor_ste, keeps continuous values."""
        levels_minus_1 = self._levels - 1
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (z.tanh() + 1) / 2.0) + 0.5
        return (
            scale * bracket - 1.0
        )  # Output in [-1, 1] to match implicit_codebook range

    def quantize(self, z):
        """
        Stochastic quantization with top-k sampling based on joint probability.

        Process:
        1. Soft bound z to continuous values in [-1, 1]
        2. Compute per-dimension probabilities against q_levels
        3. Calculate joint log probability for each codebook entry
        4. Select top-k codes or sample from them
        """
        if not self.do_sample:
            return super().quantize(z)

        bound_fn = (
            self.symmetry_preserving_bound_soft
            if self.preserve_symmetry
            else self.bound_soft
        )
        bounded_z = bound_fn(z)  # (B, N, C, D)

        B, N, C, D = bounded_z.shape
        device = z.device
        dtype = bounded_z.dtype

        # Compute per-dimension log probabilities: P(level_d | z_d) for each dimension
        log_probs_per_dim = []
        for d in range(D):
            u = bounded_z[..., d]  # (B, N, C)
            qd = self.q_levels[d]  # (Kd,)
            d2 = (u.unsqueeze(-1) - qd) ** 2  # (B, N, C, Kd)
            logits = -d2 / (self.temperature * self.temperature + 1e-10)
            log_probs = torch.log_softmax(logits, dim=-1)  # (B, N, C, Kd)
            log_probs_per_dim.append(log_probs)

        # Compute joint log probability for each codebook entry
        # P(code) = ∏_d P(level_d | z_d) = exp(∑_d log P(level_d | z_d))
        joint_log_probs = torch.zeros(
            B, N, C, self.codebook_size, device=device, dtype=dtype
        )
        for d in range(D):
            indices = self.codebook_indices[:, d]  # (codebook_size,)
            indices_expanded = indices.view(1, 1, 1, self.codebook_size).expand(
                B, N, C, self.codebook_size
            )
            selected_log_probs = torch.gather(
                log_probs_per_dim[d], dim=-1, index=indices_expanded
            )
            joint_log_probs += selected_log_probs

        # Select top-k codes with highest joint probability
        topk_log_probs, topk_code_indices = torch.topk(
            joint_log_probs, k=self.top_k, dim=-1, largest=True
        )

        if self.return_top_k:
            # Return all top-k codes
            flat_indices = topk_code_indices.reshape(-1, self.top_k)
            flat_codes = self.implicit_codebook[flat_indices]
            quantized = flat_codes.reshape(B, N, C, self.top_k, D)

            if C == 1:
                quantized = quantized.squeeze(2)  # (B, N, top_k, D)
                bounded_z_expanded = bounded_z  # (B, N, 1, D)
            else:
                raise NotImplementedError(
                    "Multiple codebooks not supported for return_top_k=True"
                )

            # Straight-through estimator
            return bounded_z_expanded + (quantized - bounded_z_expanded).detach()

        else:
            # Sample one code from top-k
            probs = torch.softmax(topk_log_probs, dim=-1)
            probs_flat = probs.reshape(-1, self.top_k)
            sampled_idx = torch.multinomial(probs_flat, num_samples=1)
            sampled_idx = sampled_idx.reshape(B, N, C, 1)

            selected_code_idx = torch.gather(
                topk_code_indices, dim=-1, index=sampled_idx
            )

            flat_selected = selected_code_idx.reshape(-1)
            flat_codes = self.implicit_codebook[flat_selected]
            quantized = flat_codes.reshape(B, N, C, D)

            # Straight-through estimator
            return bounded_z + (quantized - bounded_z).detach()

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            # codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = maybe(unpack_one)(indices, ps, "b * c")

        # if not self.keep_num_codebooks_dim and self.return_indices:
        if not self.return_top_k and self.return_indices:
            indices = maybe(rearrange)(indices, "... 1 -> ...")

        # return quantized output and indices

        return out, indices


def test():
    fsq = StochasticFSQTopK(
        levels=[8, 5, 5, 5], dim=512, return_top_k=False, do_sample=False
    )
    x = torch.randn(1, 1024, 512)
    xhat, indices = fsq(x)
    print(xhat.shape)  # (1, 1024, top_k, 512)
    print(indices.shape)  # (1, 1024, top_k,)


if __name__ == "__main__":
    test()
