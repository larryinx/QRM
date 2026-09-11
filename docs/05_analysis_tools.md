# Analysis tools

`python -m qrm.analyze` loads a checkpoint (uncompiled, EMA weights if
present), runs the full recursion on the test split, and records
per-step diagnostics. Two analyzers are available.

```bash
uv run python -m qrm.analyze \
    --ckpt_dir results/<run>/final_checkpoint \
    --analyzer trm \
    [--output_dir <dir>]            # default: <ckpt_dir>/analysis
    [--max_samples N]               # default: whole test split
    [--batch_size 768]
    [--fsq_sampling_inference true|false]   # override the checkpoint's setting
    [--save_preds]                  # trm analyzer: also dump predictions
    [--analyze_rounds 1,8,16]       # fsq_entropy analyzer: which steps to record
```

## `trm` analyzer

`TRMAnalyzer` writes one `round_XX.json` per supervision step with
token accuracy, exact accuracy, halt statistics and, with
`--save_preds`, per-sample predictions, plus a `metadata.json`. Use it
to see how accuracy evolves across the 16 recursion steps of a TRM or
TRM + FSQ checkpoint.

## `fsq_entropy` analyzer

`FSQEntropyAnalyzer` hooks the FSQ layer at the requested steps and
writes `fsq_entropy_summary.json`:

```json
{
  "config": {"fsq_levels": [8,5,5,5], "fsq_top_k": 8, "fsq_temperature": 1.0,
             "fsq_residual": {"mode": "fixed", "alpha": 0.3},
             "codebook_size": 1000, "codebook_dim": 4, "analyze_rounds": [0,7,15], ...},
  "num_samples": 1024,
  "per_round": {
    "round_01": {
      "d1_codebook_utilization": {"active_codes": 114, "codebook_size": 1000,
                                  "utilization_rate": 0.114, "top10_codes_cover_pct": 0.92,
                                  "top50_codes_cover_pct": 0.997, "verdict": "..."},
      "d2_position_entropy":     {"mean_entropy": 1.64, "answer_positions_mean": 1.64,
                                  "prompt_positions_mean": 0.0, "min_entropy": 1.46,
                                  "max_entropy": 1.80, "max_possible_entropy": 9.97, "verdict": "..."},
      "d3_topk_gap":             {"mean_top1_prob": 0.0186, "mean_gap_1_2": 0.055,
                                  "mean_gap_1_k": 0.28, "mean_topk_probs": [...]},
      "d4_per_dim_levels":       {"dims": [{"dim": 0, "levels": 8, "entropy": 0.85,
                                            "max_entropy": 3.0, "active_levels": 8,
                                            "histogram": [73065, 2988, ...]}, ...],
                                  "effective_codebook_size": 1000},
      "d5_quantization_error":   {"mean_mse": 0.0023, "mean_cosine_similarity": 0.998}
    },
    "round_08": {...}, "round_16": {...}
  }
}
```

What each block measures and how to read it is covered in
[06_fsq_entropy_analysis.md](06_fsq_entropy_analysis.md). The
`verdict` strings are threshold labels from the original collapse
hypothesis; they are descriptive, not a quality judgement.

## Comparison reports

`scripts/analysis/fsq_entropy_report.py` turns several summary files
into markdown tables. It reads a JSON manifest:

```json
{
  "round": "round_16",
  "experiments": {
    "ATT res0.3 F/F": {"summary": "analysis/fsq_entropy/att_res03/fsq_entropy_summary.json",
                        "exact_acc": 0.7694},
    "ATT F/F":        {"summary": "analysis/fsq_entropy/att_ff/fsq_entropy_summary.json",
                        "exact_acc": 0.3238}
  },
  "trajectories": {
    "ATT res0.3 F/F": {"Early": "analysis/fsq_entropy_training/att_res03/ep01/fsq_entropy_summary.json",
                        "Mid":   "analysis/fsq_entropy_training/att_res03/ep05/fsq_entropy_summary.json",
                        "Final": "analysis/fsq_entropy/att_res03/fsq_entropy_summary.json"}
  }
}
```

```bash
uv run python scripts/analysis/fsq_entropy_report.py \
    --manifest scripts/analysis/fsq_entropy_manifest.example.json \
    --output analysis/fsq_entropy_report.md
```

`experiments` produces the per-step comparison tables (D1 to D5 and the
residual configuration); `trajectories` (optional) produces the
training-trajectory and within-inference tables. Missing files are
skipped with a warning.

`scripts/analysis/run_fsq_entropy.sh` and `scripts/analysis/run_trm_analysis.sh`
are minimal wrappers that show the intended invocation for one
checkpoint.

## Dataset checks

`scripts/analysis/check_dataset.py` validates a built dataset directory:
it prints array shapes and dtypes, checks that puzzle and group indices
are consistent, and for Sudoku verifies that every augmented row is a
valid puzzle/solution pair.

```bash
uv run python scripts/analysis/check_dataset.py data/sudoku-extreme-1k-aug-1000
```
