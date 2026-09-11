"""Unit tests for StochasticFSQ module."""

import pytest
import torch

from qrm.layers.stochastic_fsq import StochasticFSQ


class TestStochasticFSQBasics:
    """Test basic functionality of StochasticFSQ."""

    def test_init(self):
        """Test initialization with default parameters."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        assert fsq.codebook_dim == 4
        assert fsq.codebook_size == 8 * 5 * 5 * 5  # 1000
        assert fsq.top_k == 8
        assert fsq.dim == 512

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        fsq = StochasticFSQ(
            levels=[4, 4, 4],
            dim=256,
            top_k=4,
            temperature=0.5,
            projection_has_bias=True,
        )

        assert fsq.codebook_dim == 3
        assert fsq.codebook_size == 64
        assert fsq.top_k == 4
        assert fsq.temperature == 0.5

    def test_large_codebook(self):
        """Test that large codebook sizes work (with warning)."""
        # 10000 is within limits
        fsq = StochasticFSQ(levels=[10, 10, 10], dim=512)
        assert fsq.codebook_size == 1000


class TestImplicitCodebook:
    """Test implicit codebook generation."""

    def test_codebook_shape(self):
        """Test that implicit codebook has correct shape."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        assert fsq.implicit_codebook.shape == (1000, 4)

    def test_codebook_range(self):
        """Test that codebook values are in [-1, 1]."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        assert fsq.implicit_codebook.min() >= -1.0
        assert fsq.implicit_codebook.max() <= 1.0

    def test_codebook_unique_entries(self):
        """Test that all codebook entries are unique."""
        fsq = StochasticFSQ(levels=[4, 4, 4], dim=256)

        # Each row should be unique
        unique_rows = torch.unique(fsq.implicit_codebook, dim=0)
        assert unique_rows.shape[0] == fsq.codebook_size

    def test_per_dimension_levels(self):
        """Test that each dimension has correct number of unique levels."""
        levels = [8, 5, 5, 5]
        fsq = StochasticFSQ(levels=levels, dim=512)

        for d in range(len(levels)):
            unique_vals = fsq.implicit_codebook[:, d].unique()
            assert (
                len(unique_vals) == levels[d]
            ), f"Dimension {d} has wrong number of levels"


class TestBoundSoft:
    """Test bound_soft function."""

    def test_output_range(self):
        """Test that bound_soft output is approximately in [-1, 1] range."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        # Test with various inputs
        z = torch.randn(2, 100, 4) * 10  # Large values
        bounded = fsq.bound_soft(z)

        # FSQ bound_soft may slightly exceed [-1, 1] due to offset handling
        # for even number of levels. Allow small tolerance.
        assert bounded.min() >= -1.1
        assert bounded.max() <= 1.1

    def test_shape_preservation(self):
        """Test that bound_soft preserves shape."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        z = torch.randn(2, 100, 4)
        bounded = fsq.bound_soft(z)

        assert bounded.shape == z.shape

    def test_differentiability(self):
        """Test that bound_soft is differentiable."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        z = torch.randn(2, 100, 4, requires_grad=True)
        bounded = fsq.bound_soft(z)
        loss = bounded.sum()
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()


class TestJointLogProbs:
    """Test joint log probability computation."""

    def test_output_shape(self):
        """Test joint_log_probs output shape."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        bounded_z = torch.randn(2, 100, 4)
        log_probs = fsq.compute_joint_log_probs(bounded_z)

        assert log_probs.shape == (2, 100, 1000)

    def test_log_probs_finite(self):
        """Test that log probabilities are finite."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        bounded_z = torch.randn(2, 100, 4)
        log_probs = fsq.compute_joint_log_probs(bounded_z)

        assert torch.isfinite(log_probs).all()

    def test_probs_sum_to_one(self):
        """Test that probabilities approximately sum to 1."""
        fsq = StochasticFSQ(
            levels=[4, 4], dim=256
        )  # Small codebook for numerical stability

        bounded_z = torch.randn(2, 10, 2)
        log_probs = fsq.compute_joint_log_probs(bounded_z)
        probs = torch.exp(log_probs)

        # Sum should be close to 1 (allowing for numerical errors)
        prob_sum = probs.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3)

    def test_temperature_effect(self):
        """Test that lower temperature produces sharper distribution."""
        fsq_low_temp = StochasticFSQ(levels=[4, 4], dim=256, temperature=0.1)
        fsq_high_temp = StochasticFSQ(levels=[4, 4], dim=256, temperature=2.0)

        bounded_z = torch.randn(1, 1, 2)

        log_probs_low = fsq_low_temp.compute_joint_log_probs(bounded_z)
        log_probs_high = fsq_high_temp.compute_joint_log_probs(bounded_z)

        # Low temperature should have higher max probability
        max_prob_low = torch.exp(log_probs_low).max()
        max_prob_high = torch.exp(log_probs_high).max()

        assert max_prob_low > max_prob_high

    def test_codebook_indices_alignment(self):
        """Test that codebook_indices correctly maps to implicit_codebook values.

        This is critical for joint probability computation:
        - codebook_indices[i, d] should be the index of implicit_codebook[i, d] in q_levels[d]
        - i.e., q_levels[d][codebook_indices[i, d]] == implicit_codebook[i, d]
        """
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        # Verify alignment for all codebook entries
        for code_idx in range(fsq.codebook_size):
            code_values = fsq.implicit_codebook[code_idx]  # [D]
            level_indices = fsq.codebook_indices[code_idx]  # [D]

            for d in range(fsq.codebook_dim):
                reconstructed_value = fsq.q_levels[d][level_indices[d]]
                assert torch.isclose(code_values[d], reconstructed_value), (
                    f"Alignment mismatch at code_idx={code_idx}, dim={d}: "
                    f"implicit_codebook={code_values[d].item()}, "
                    f"reconstructed={reconstructed_value.item()}"
                )

    def test_joint_prob_highest_for_nearest_code(self):
        """Test that the nearest codebook entry has the highest probability.

        When input z exactly matches a codebook entry, that entry should have
        the highest probability (with low temperature for numerical stability).
        """
        fsq = StochasticFSQ(levels=[4, 4], dim=256, temperature=0.1)

        # Use exact codebook values as input
        for code_idx in [0, 5, 10, 15]:
            # Get the exact code value
            exact_code = (
                fsq.implicit_codebook[code_idx].unsqueeze(0).unsqueeze(0)
            )  # [1, 1, D]

            log_probs = fsq.compute_joint_log_probs(exact_code)  # [1, 1, codebook_size]
            predicted_idx = log_probs.argmax(dim=-1).item()

            assert (
                predicted_idx == code_idx
            ), f"Expected code_idx={code_idx} to have highest prob, got {predicted_idx}"


class TestForward:
    """Test forward pass."""

    def test_output_shapes_return_top_k(self):
        """Test output shapes when return_top_k=True."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)

        x = torch.randn(2, 100, 512)
        out, indices = fsq(x, return_top_k=True)

        assert out.shape == (2, 8, 100, 512)
        assert indices.shape == (2, 8, 100)

    def test_output_shapes_single_greedy(self):
        """Test output shapes when return_top_k=False, do_sampling=False."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)

        x = torch.randn(2, 100, 512)
        out, indices = fsq(x, return_top_k=False, do_sampling=False)

        assert out.shape == (2, 1, 100, 512)
        assert indices.shape == (2, 1, 100)

    def test_output_shapes_single_sampling(self):
        """Test output shapes when return_top_k=False, do_sampling=True."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)

        x = torch.randn(2, 100, 512)
        out, indices = fsq(x, return_top_k=False, do_sampling=True)

        assert out.shape == (2, 1, 100, 512)
        assert indices.shape == (2, 1, 100)

    def test_indices_valid_range(self):
        """Test that indices are in valid range."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)

        x = torch.randn(2, 100, 512)
        _, indices = fsq(x, return_top_k=True)

        assert indices.min() >= 0
        assert indices.max() < fsq.codebook_size

    def test_greedy_deterministic(self):
        """Test that greedy mode is deterministic."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)
        fsq.eval()

        x = torch.randn(2, 100, 512)

        out1, idx1 = fsq(x, return_top_k=False, do_sampling=False)
        out2, idx2 = fsq(x, return_top_k=False, do_sampling=False)

        assert torch.equal(idx1, idx2)
        assert torch.allclose(out1, out2)

    def test_sampling_stochastic(self):
        """Test that sampling mode produces different results."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)
        fsq.eval()

        x = torch.randn(2, 100, 512)

        # Run multiple times
        all_same = True
        _, idx1 = fsq(x, return_top_k=False, do_sampling=True)
        for _ in range(5):
            _, idx2 = fsq(x, return_top_k=False, do_sampling=True)
            if not torch.equal(idx1, idx2):
                all_same = False
                break

        # Should not always be the same (with high probability)
        assert not all_same, "Sampling should produce different results"

    def test_gradient_flow(self):
        """Test that gradients flow through forward pass."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)

        x = torch.randn(2, 100, 512, requires_grad=True)
        out, _ = fsq(x, return_top_k=True)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_sampling(self):
        """Test gradients with sampling mode."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=8)

        x = torch.randn(2, 100, 512, requires_grad=True)
        out, _ = fsq(x, return_top_k=False, do_sampling=True)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestIndexConversion:
    """Test index to code and code to index conversion."""

    def test_indices_to_codes_shape(self):
        """Test indices_to_codes output shape."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        indices = torch.randint(0, fsq.codebook_size, (2, 100))
        codes = fsq.indices_to_codes(indices)

        assert codes.shape == (2, 100, 4)

    def test_codes_to_indices_roundtrip(self):
        """Test that codes_to_indices is inverse of indices_to_codes."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        indices = torch.randint(0, fsq.codebook_size, (2, 100))
        codes = fsq.indices_to_codes(indices)
        recovered_indices = fsq.codes_to_indices(codes)

        assert torch.equal(indices, recovered_indices)


class TestDeviceAndDtype:
    """Test device and dtype handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512).cuda()

        x = torch.randn(2, 100, 512).cuda()
        out, indices = fsq(x, return_top_k=True)

        assert out.device.type == "cuda"
        assert indices.device.type == "cuda"

    def test_bfloat16_forward(self):
        """Test forward pass with bfloat16."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)
        fsq = fsq.to(torch.bfloat16)

        x = torch.randn(2, 100, 512, dtype=torch.bfloat16)
        out, indices = fsq(x, return_top_k=True)

        assert out.dtype == torch.bfloat16


class TestEdgeCases:
    """Test edge cases."""

    def test_single_batch_single_seq(self):
        """Test with batch=1, seq=1."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        x = torch.randn(1, 1, 512)
        out, indices = fsq(x, return_top_k=True)

        assert out.shape == (1, 8, 1, 512)

    def test_small_top_k(self):
        """Test with top_k=1."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, top_k=1)

        x = torch.randn(2, 100, 512)
        out, indices = fsq(x, return_top_k=True)

        assert out.shape == (2, 1, 100, 512)

    def test_large_batch(self):
        """Test with large batch size."""
        fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512)

        x = torch.randn(64, 900, 512)
        out, indices = fsq(x, return_top_k=True)

        assert out.shape == (64, 8, 900, 512)


def test_comparison_with_greedy_quantization():
    """Test that greedy selection matches deterministic FSQ behavior."""
    fsq = StochasticFSQ(levels=[8, 5, 5, 5], dim=512, temperature=0.01)  # Very low temp

    x = torch.randn(2, 10, 512)
    out_greedy, _ = fsq(x, return_top_k=False, do_sampling=False)
    out_topk, _ = fsq(x, return_top_k=True)

    # With very low temperature, top-1 should dominate
    # Greedy output should be close to first candidate from top-k
    # FSQ output shape: [B, M, N, D] -> greedy is [B, 1, N, D], topk is [B, M, N, D]
    assert torch.allclose(out_greedy.squeeze(1), out_topk[:, 0, :, :], atol=1e-5)


class TestFSQCompatibility:
    """Test compatibility with vector_quantize_pytorch.FSQ."""

    def test_codebook_matches_original_fsq(self):
        """Test that our implicit_codebook matches the original FSQ."""
        from vector_quantize_pytorch import FSQ as OriginalFSQ

        levels = [8, 5, 5, 5]
        our_fsq = StochasticFSQ(levels=levels, dim=512)
        orig_fsq = OriginalFSQ(levels=levels, dim=512)

        # Compare implicit codebooks
        assert our_fsq.implicit_codebook.shape == orig_fsq.implicit_codebook.shape
        assert torch.allclose(
            our_fsq.implicit_codebook, orig_fsq.implicit_codebook, atol=1e-6
        ), "Implicit codebook mismatch!"

    def test_codes_to_indices_matches_original(self):
        """Test that codes_to_indices gives same results as original FSQ."""
        from vector_quantize_pytorch import FSQ as OriginalFSQ

        levels = [8, 5, 5, 5]
        our_fsq = StochasticFSQ(levels=levels, dim=512)
        orig_fsq = OriginalFSQ(levels=levels, dim=512)

        # Test with all codebook entries
        codes = our_fsq.implicit_codebook
        our_indices = our_fsq.codes_to_indices(codes)
        orig_indices = orig_fsq.codes_to_indices(codes)

        assert torch.equal(our_indices, orig_indices), "codes_to_indices mismatch!"

    def test_indices_to_codes_matches_original(self):
        """Test that indices_to_codes gives same results as original FSQ."""
        from vector_quantize_pytorch import FSQ as OriginalFSQ

        levels = [8, 5, 5, 5]
        our_fsq = StochasticFSQ(levels=levels, dim=512)
        orig_fsq = OriginalFSQ(levels=levels, dim=512)

        # Test with random indices
        indices = torch.randint(0, our_fsq.codebook_size, (10, 20))
        our_codes = our_fsq.indices_to_codes(indices)
        orig_codes = orig_fsq._indices_to_codes(indices)

        assert torch.allclose(
            our_codes, orig_codes, atol=1e-6
        ), "indices_to_codes mismatch!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
