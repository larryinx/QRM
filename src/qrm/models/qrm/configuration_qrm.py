"""QRM Configuration.

QRM (Quantized Recursive Model) extends TRM with:
1. Stochastic FSQ quantization
2. MCTS tree search over FSQ candidates
3. Multi-candidate training with weighted loss

Note:
- halt_max_steps = max tree depth AND inference iteration budget
- num_iterations = training iteration budget (gradient_accumulation_steps)
"""

from qrm.models.trm.configuration_trm import TRMConfig


class QRMConfig(TRMConfig):
    """QRM configuration class, extends TRM configuration.

    Design principles:
    1. All TRM parameters preserved (halt_max_steps = max tree depth)
    2. FSQ-related configuration added
    3. Loss-related configuration added
    4. MCTS configuration: search_rule, c_uct, c_puct, terminal_selection_mode
    """

    model_type = "qrm"

    def __init__(
        self,
        # === FSQ Configuration ===
        fsq_levels: tuple[int, ...] = (8, 5, 5, 5),  # e.g., [8, 5, 5, 5]
        fsq_top_k: int = 8,  # Candidates for training & MCTS expand width
        fsq_temperature: float = 1.0,  # Temperature for probability computation
        fsq_residual_mode: str = "fixed",  # "fixed" | "learned_scalar" (no "attention" for QRM)
        fsq_residual_weight: float = 1.0,  # α init; 1.0 = pure quantization (backward compat)
        fsq_pre_norm: bool = False,  # Normalize project_in output before tanh
        # === Inference Configuration ===
        do_sampling: bool = False,  # Whether to sample during inference
        # === Loss Configuration (Milestone 1 & MCTS shared) ===
        # Tree Loss
        lambda_tree: float = 1.0,
        alpha: float = 0.5,  # Depth penalty coefficient (for MCTS)
        # Reconstruction Loss
        lambda_recon: float = 0.1,  # Weight for MSE(z_continuous, z_quantized)
        # Diversity Loss
        lambda_div: float = 0.01,  # Weight for diversity loss
        diversity_beta: float = 1.0,  # Distance exponent: 1.0=L2, 2.0=L2²
        diversity_seq_aggregation: str = "mean",  # "mean" or "sum"
        # Reward Configuration
        em_weight: float = 0.0,  # Exact match weight (disabled by default)
        token_acc_weight: float = 1.0,
        # === MCTS Configuration ===
        # MCTS uses halt_max_steps as max_depth, fsq_top_k as expand_width
        num_iterations: int = 32,  # Training iteration budget (gradient_accumulation_steps)
        search_rule: str = "uct",  # "uct" | "puct" (UCT is primary; PUCT kept for ablation)
        c_uct: float = 1.414,  # Classic UCB1 exploration constant (~sqrt(2))
        c_puct: float = 1.0,  # PUCT exploration coefficient
        q_normalize: bool = True,  # Clip Q(s,a) into [0, 1] before use in selection
        terminal_selection_mode: str = "skip_saturated",  # "replay_parent" | "skip_saturated"
        **kwargs,
    ):
        # Enable FSQ by default for QRM
        kwargs.setdefault("use_fsq", True)

        # Pass FSQ params explicitly to TRMConfig (they're set in the if use_fsq: branch)
        super().__init__(
            fsq_levels=fsq_levels,
            fsq_top_k=fsq_top_k,
            fsq_temperature=fsq_temperature,
            fsq_residual_mode=fsq_residual_mode,
            fsq_residual_weight=fsq_residual_weight,
            fsq_pre_norm=fsq_pre_norm,
            **kwargs,
        )

        # Validate: attention mode not supported for QRM (M > 1 candidates)
        if fsq_residual_mode == "attention":
            raise ValueError(
                "fsq_residual_mode='attention' is not supported for QRM. "
                "QRM returns M>1 candidates; use 'fixed' or 'learned_scalar' instead."
            )
        if search_rule not in {"uct", "puct"}:
            raise ValueError(
                f"Unknown search_rule={search_rule!r}. Use 'uct' or 'puct'."
            )
        if terminal_selection_mode not in {"replay_parent", "skip_saturated"}:
            raise ValueError(
                "Unknown terminal_selection_mode="
                f"{terminal_selection_mode!r}. Use 'replay_parent' or 'skip_saturated'."
            )

        # Override FSQ params (TRMConfig already set them, but ensure consistency)
        self.fsq_levels = list(fsq_levels) if fsq_levels else [8, 5, 5, 5]
        self.fsq_top_k = fsq_top_k
        self.fsq_temperature = fsq_temperature

        # Inference
        self.do_sampling = do_sampling

        # Loss
        self.lambda_tree = lambda_tree
        self.alpha = alpha
        self.lambda_recon = lambda_recon
        self.lambda_div = lambda_div
        self.diversity_beta = diversity_beta
        self.diversity_seq_aggregation = diversity_seq_aggregation
        self.em_weight = em_weight
        self.token_acc_weight = token_acc_weight

        # MCTS: max_depth = halt_max_steps, expand_width = fsq_top_k
        self.num_iterations = num_iterations
        self.search_rule = search_rule
        self.c_uct = c_uct
        self.c_puct = c_puct
        self.q_normalize = q_normalize
        self.terminal_selection_mode = terminal_selection_mode


if __name__ == "__main__":
    # Sudoku config for Milestone 1
    sudoku_qrm_config = QRMConfig(
        seq_len=81,
        vocab_size=11,
        num_puzzle_identifiers=1,
        L_layers=2,
        H_cycles=3,
        L_cycles=6,
        halt_max_steps=16,  # Fixed inference steps (reused from TRM)
        # QRM specific
        fsq_levels=[8, 5, 5, 5],
        fsq_top_k=8,
    )
    print("Sudoku QRM Config:")
    print(f"  model_type: {sudoku_qrm_config.model_type}")
    print(f"  use_fsq: {sudoku_qrm_config.use_fsq}")
    print(f"  fsq_levels: {sudoku_qrm_config.fsq_levels}")
    print(f"  fsq_top_k: {sudoku_qrm_config.fsq_top_k}")
    print(
        f"  halt_max_steps (fixed inference steps): {sudoku_qrm_config.halt_max_steps}"
    )

    # ARC-AGI config for later
    arc_qrm_config = QRMConfig(
        seq_len=900,
        vocab_size=12,
        num_puzzle_identifiers=1200,
        L_layers=2,
        H_cycles=3,
        L_cycles=4,
        halt_max_steps=16,  # Fixed inference steps
        # QRM specific
        fsq_levels=[8, 5, 5, 5],
        fsq_top_k=8,
    )
    print("\nARC-AGI QRM Config:")
    print(f"  model_type: {arc_qrm_config.model_type}")
    print(f"  seq_len: {arc_qrm_config.seq_len}")
    print(f"  halt_max_steps (fixed inference steps): {arc_qrm_config.halt_max_steps}")
