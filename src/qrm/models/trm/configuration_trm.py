from transformers import PretrainedConfig


class TRMConfig(PretrainedConfig):
    """TRM configuration class."""

    model_type = "trm"

    def __init__(
        self,
        # Basic config
        batch_size: int = 192,  # Required by CastedSparseEmbedding
        # 900 for ARC-AGI-1/2, 30x30 grid
        # 81 for sudoku, 9x9 grid
        # 900 for Maze-30x30, 30x30 grid
        seq_len: int = 81,
        # 12 for ARC-AGI-1/2, PAD + EOS + "0"..."9" (10 colors)
        # 11 for sudoku, PAD + "0"..."9"
        # 6 for Maze-30x30, PAD + CHARSET("# SGo")
        vocab_size: int = 11,
        # Model architecture
        hidden_size: int = 512,
        num_heads: int = 8,
        expansion: float = 4.0,
        H_layers: int = 0,  # Currently unused
        L_layers: int = 2,
        # Recursion config
        H_cycles: int = 3,
        L_cycles: int = 6,
        # Regularization
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        # Halt/ACT config
        halt_exploration_prob: float = 0.1,
        halt_max_steps: int = 16,
        # Embedding config
        puzzle_emb_ndim: int = 512,  # ${.hidden_size}
        puzzle_emb_len: int = 16,
        puzzle_emb_rank: int = None,  # Low-rank dim, None = full rank (no projection)
        # ARC-AGI-1/2, ~1200+ unique identifiers
        # Sudoku & Maze-30x30, 1 shared identifier
        num_puzzle_identifiers: int = 1,
        # Other
        pos_encodings: str = "rope",
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        no_ACT_continue: bool = True,
        use_fsq: bool = False,
        fsq_levels: tuple[int, ...] = (8, 5, 5, 5),
        # WARNING: With top_k=1, do_sampling has NO effect (sampling from 1 candidate
        # always picks that candidate). To make fsq_sampling_training/inference meaningful,
        # increase top_k (e.g., 8). return_top_k=False in TRM guarantees M=1 output
        # regardless of top_k, so top_k>1 is safe here.
        fsq_top_k: int = 8,
        fsq_temperature: float = 1.0,
        fsq_sampling_training: bool = False,
        fsq_sampling_inference: bool = False,
        fsq_residual_mode: str = "fixed",  # "fixed" | "learned_scalar" (| "learned_vector" TODO)
        fsq_residual_weight: float = 1.0,  # α init; 1.0 = pure quantization (backward compat)
        fsq_pre_norm: bool = False,  # Normalize project_in output before tanh (prevents bimodal collapse)
        fsq_recon_weight: float = 0.0,  # MSE(z_quantized, z_continuous) weight; 0.0 = disabled
        causal: bool = False,
        # Weighted exact match loss
        weighted_em_loss: bool = False,
        weighted_em_weight: float = 1.0,
        # Training config
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.halt_exploration_prob = halt_exploration_prob
        self.halt_max_steps = halt_max_steps
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.puzzle_emb_len = puzzle_emb_len
        self.puzzle_emb_rank = puzzle_emb_rank
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.pos_encodings = pos_encodings
        self.forward_dtype = forward_dtype
        self.mlp_t = mlp_t
        self.no_ACT_continue = no_ACT_continue
        self.use_fsq = use_fsq
        self.causal = causal
        self.weighted_em_loss = weighted_em_loss
        self.weighted_em_weight = weighted_em_weight

        if use_fsq:
            self.fsq_levels = list(fsq_levels) if fsq_levels else [8, 5, 5, 5]
            self.fsq_top_k = fsq_top_k
            self.fsq_temperature = fsq_temperature
            self.fsq_sampling_training = fsq_sampling_training
            self.fsq_sampling_inference = fsq_sampling_inference
            self.fsq_residual_mode = fsq_residual_mode
            self.fsq_residual_weight = fsq_residual_weight
            self.fsq_pre_norm = fsq_pre_norm
            self.fsq_recon_weight = fsq_recon_weight
            if fsq_residual_mode not in ("fixed", "learned_scalar", "attention"):
                raise ValueError(
                    f"fsq_residual_mode must be 'fixed', 'learned_scalar', or 'attention', got '{fsq_residual_mode}'"
                )
            # Note: top_k > 1 is allowed because TRM always uses return_top_k=False,
            # so output M=1 regardless. top_k controls the sampling pool size.
        else:
            self.fsq_levels = None
            self.fsq_top_k = None
            self.fsq_temperature = None
            self.fsq_sampling_training = None
            self.fsq_sampling_inference = None
            self.fsq_residual_mode = None
            self.fsq_residual_weight = None
            self.fsq_pre_norm = None
            self.fsq_recon_weight = None

if __name__ == "__main__":
    arc_agi_1_config = TRMConfig(
        seq_len=900,
        vocab_size=12,
        num_puzzle_identifiers=1200,  # depends
        L_layers=2,
        H_cycles=3,
        L_cycles=4,
    )

    arc_agi_2_config = TRMConfig(
        seq_len=900,
        vocab_size=12,
        num_puzzle_identifiers=1200,  # depends
        L_layers=2,
        H_cycles=3,
        L_cycles=4,
    )

    sudoku_ext_mlp_t_config = TRMConfig(
        seq_len=81,
        vocab_size=11,
        num_puzzle_identifiers=1,
        mlp_t=True,
        pos_encodings="none",
        L_layers=2,
        H_cycles=3,
        L_cycles=6,
    )

    sudoku_ext_att_config = TRMConfig(
        seq_len=81,
        vocab_size=11,
        num_puzzle_identifiers=1,
        L_layers=2,
        H_cycles=3,
        L_cycles=6,
    )

    maze_hard_config = TRMConfig(
        seq_len=900,
        vocab_size=6,
        num_puzzle_identifiers=1,
        L_layers=2,
        H_cycles=3,
        L_cycles=4,
    )
