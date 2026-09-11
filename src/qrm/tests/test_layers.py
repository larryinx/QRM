import torch

from qrm.layers.common import trunc_normal_init_
from qrm.layers.sparse_embedding import CastedSparseEmbedding


class TestTruncNormalInit:
    def test_zero_std(self):
        """Test that std=0 results in all zeros."""
        tensor = torch.empty(10, 10)
        trunc_normal_init_(tensor, std=0)
        assert torch.allclose(tensor, torch.zeros_like(tensor))

    def test_distribution_bounds(self):
        """Test truncation boundaries."""
        import math

        torch.manual_seed(42)
        tensor = torch.empty(10000)
        std, lower, upper = 1.0, -2.0, 2.0
        trunc_normal_init_(tensor, std=std, lower=lower, upper=upper)

        # Compute comp_std (same as trunc_normal_init_ internal logic)
        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower**2)
        pdf_l = c * math.exp(-0.5 * upper**2)
        comp_std = std / math.sqrt(
            1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
        )

        # Actual truncation bounds are [lower * comp_std, upper * comp_std]
        assert tensor.min() >= lower * comp_std
        assert tensor.max() <= upper * comp_std


class TestCastedSparseEmbedding:
    def test_buffer_structure(self):
        emb = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=32,
            init_std=0.1,
            cast_to=torch.bfloat16,
        )

        # weights should be Buffer, not Parameter
        assert "weights" in dict(emb.named_buffers())
        assert "weights" not in dict(emb.named_parameters())

        # local_weights should have requires_grad
        assert emb.local_weights.requires_grad

    def test_training_mode(self):
        emb = CastedSparseEmbedding(100, 64, 4, 0.1, torch.float32)
        emb.train()

        ids = torch.tensor([1, 5, 10, 20])
        _ = emb(ids)

        # Should fill local_ids
        assert torch.equal(emb.local_ids, ids)


if __name__ == "__main__":
    test = TestTruncNormalInit()
    test.test_zero_std()
    # test.test_distribution_bounds()

    test = TestCastedSparseEmbedding()
    test.test_training_mode()
    test.test_buffer_structure()
