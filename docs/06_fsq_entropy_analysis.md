# FSQ codebook analysis: does quantization collapse?

This note records the diagnostic study we ran on TRM + FSQ checkpoints
(Sudoku-Extreme and Maze-Hard) after observing that FSQ hurt Sudoku
accuracy but not Maze accuracy. The original hypothesis was *codebook
entropy collapse*: the model uses only a handful of codes, so the discrete
bottleneck destroys information. The data said the opposite, and the
outcome shaped two design choices that are now defaults in the codebase
(the residual mix `fsq_residual_weight` and the optional `fsq_pre_norm`).

All numbers below come from `python -m qrm.analyze --analyzer fsq_entropy`
run on the test split with `--fsq_sampling_inference false`, so every
checkpoint is measured under greedy quantization regardless of how it
was trained. See [analysis_tools.md](analysis_tools.md) for how to
reproduce the tables.

## 1. What is measured

`FSQEntropyAnalyzer` (`src/qrm/analyzers/fsq_entropy_analyzer.py`) runs
the full 16-step recursion on a batch of test puzzles, hooks
`StochasticFSQ.forward(..., return_diagnostics=True)` at selected steps,
strips the puzzle-embedding prefix, and writes one
`fsq_entropy_summary.json` per checkpoint with five diagnostics per
analysed step:

| Key | Diagnostic | Main fields |
|---|---|---|
| `d1_codebook_utilization` | how many of the `prod(levels)` codes are ever selected | `active_codes`, `utilization_rate`, `top10_codes_cover_pct`, `top50_codes_cover_pct` |
| `d2_position_entropy` | entropy (bits) of the selected code at each sequence position, across samples | `mean_entropy`, `max_possible_entropy = log2(codebook_size)` |
| `d3_topk_gap` | how peaked the Gibbs distribution over codes is | `mean_top1_prob`, `mean_gap_1_2`, `mean_gap_1_k` |
| `d4_per_dim_levels` | per-codebook-dimension histogram of selected levels | `entropy`, `active_levels`, `histogram` per dim, `effective_codebook_size` |
| `d5_quantization_error` | distance between the soft-bounded latent and the selected code | `mean_mse`, `mean_cosine_similarity` |

The verdict strings (`SEVERE_COLLAPSE`, `MODERATE_COLLAPSE`,
`MILD_UNDERUTILIZATION`, `HEALTHY`) are threshold labels on utilization
(< 0.10, < 0.30, < 0.50) and on the entropy ratio (< 0.30, < 0.50). They
encode the *original* hypothesis and should be read as descriptive tags,
not as judgements, for the reason explained next.

## 2. Experiments

All runs use `fsq_levels=[8,5,5,5]` (1000 codes) unless marked `[85555]`
(`[8,5,5,5,5]`, 5000 codes). `F/F` and `T/T` are
`fsq_sampling_training` / `fsq_sampling_inference`. `res0.3` means
`fsq_residual_mode=fixed, fsq_residual_weight=0.3`, i.e.
`z_L <- 0.3 * quantized + 0.7 * continuous`. `EM(w)` means
`weighted_em_loss=True, weighted_em_weight=w`.

| Run | Task | Model | Exact accuracy |
|---|---|---|---|
| Maze ATT T/T | Maze-Hard | TRM-Att | 0.8539 |
| ATT [8555] F/F | Sudoku | TRM-Att | 0.3238 |
| ATT [8555] T/T | Sudoku | TRM-Att | 0.4611 |
| MLP [8555] F/F | Sudoku | TRM-MLP | 0.6183 |
| MLP [8555] res0.3 F/F | Sudoku | TRM-MLP | 0.8437 |
| ATT [8555] res0.3 F/F | Sudoku | TRM-Att | 0.7694 |
| ATT [85555] F/F | Sudoku | TRM-Att | 0.0865 |
| MLP [85555] F/F | Sudoku | TRM-MLP | 0.6616 |
| MLP [85555] res0.5 T/T | Sudoku | TRM-MLP | 0.8074 |
| ATT EM(0.5) F/F | Sudoku | TRM-Att | 0.5363 |
| ATT EM(1.0) F/F | Sudoku | TRM-Att | 0.7111 |

For reference, TRM without FSQ trained under the same budget reaches
0.7748 (Att) and 0.8129 (MLP) on Sudoku and 0.8506 (Att) on Maze; the
full-budget replication numbers are higher (see
[07_experiments.md](07_experiments.md)).

## 3. Final-step diagnostics (recursion step 16)

### Codebook utilization and position entropy

| Run | Exact acc | Active codes | Top-10 cover | Mean entropy / max |
|---|---|---|---|---|
| Maze ATT T/T | 0.8539 | 114 / 1000 | 95.5% | 2.08 / 9.97 |
| ATT [8555] res0.3 F/F | 0.7694 | 114 / 1000 | 91.8% | 1.64 / 9.97 |
| MLP [8555] res0.3 F/F | 0.8437 | 164 / 1000 | 92.5% | 2.88 / 9.97 |
| MLP [85555] res0.5 T/T | 0.8074 | 1418 / 5000 | 33.5% | 6.55 / 12.29 |
| ATT EM(1.0) F/F | 0.7111 | 823 / 1000 | 54.0% | 5.86 / 9.97 |
| MLP [85555] F/F | 0.6616 | 2211 / 5000 | 36.4% | 6.85 / 12.29 |
| MLP [8555] F/F | 0.6183 | 842 / 1000 | 34.9% | 6.71 / 9.97 |
| ATT EM(0.5) F/F | 0.5363 | 901 / 1000 | 53.9% | 6.13 / 9.97 |
| ATT [8555] T/T | 0.4611 | 986 / 1000 | 39.8% | 6.87 / 9.97 |
| ATT [8555] F/F | 0.3238 | 851 / 1000 | 31.0% | 7.23 / 9.97 |
| ATT [85555] F/F | 0.0865 | 2137 / 5000 | 18.1% | 8.23 / 12.29 |

The ordering is the inverse of the hypothesis. The three best runs use
about 11 to 16 percent of the codebook and have position entropy near
2 bits; the failing runs spread over most of the codebook and have
entropy of 7 bits or more. Low codebook usage is a sign that the model
has settled on a small, stable set of discrete states, not a failure
mode.

### Peakedness and reconstruction

`mean_top1_prob` is between 0.004 and 0.019 for every run. At
`fsq_temperature=1.0` the Gibbs distribution over 1000 or 5000 codes is
almost flat, so the `d3` diagnostic carries little signal; sampling
(`T/T`) therefore behaves like noise injection rather than informed
exploration at this temperature. Cosine similarity between the soft
latent and the chosen code is 0.985 to 0.998 everywhere; the best
run (ATT res0.3) has the lowest MSE, 0.0023.

### Per-dimension level histograms

`effective_codebook_size` is 1000 or 5000 in every run because every
level of every dimension is hit at least once. The histograms are the
informative part. For ATT res0.3 at step 16 (dimension 1, five levels):

```
level:  -1     -0.5    0     0.5    1
count:  67820  2454    1547  2406   8717   ->  81.8%  3.0%  1.9%  2.9%  10.5%
```

Good models place almost all mass on the two endpoint levels of each
dimension. Maze does the same (dimension 3: 89.9% and 8.2% at the two
ends). Failing runs are close to uniform across levels.

The mechanism is `bound_soft`, which maps the projected latent through
`tanh` before rounding. When the pre-tanh activations have standard
deviation of about 2 or more, the induced distribution over levels is
U-shaped regardless of the level count. A well-trained model that pushes
its activations to saturation ends up using each dimension as a binary
feature. With four dimensions that is roughly 16 effective codes,
which matches the observed utilization of 100 to 160 codes once you
allow some spread.

## 4. Training trajectories

The same diagnostics on early and mid checkpoints (step 16, test split):

| Run | Utilization early -> mid -> final | Entropy early -> final |
|---|---|---|
| Maze ATT T/T | 38.1% -> 16.4% -> 11.4% | 2.85 -> 2.08 |
| MLP res0.3 F/F | 63.5% -> 22.3% -> 16.4% | 4.20 -> 2.88 |
| ATT res0.3 F/F | 8.6% -> 9.9% -> 11.4% | 1.76 -> 1.64 |
| MLP [85555] res0.5 T/T | 18.6% -> 22.3% -> 28.4% | 4.93 -> 6.55 |
| ATT EM(1.0) F/F | 6.2% -> 43.9% -> 82.3% | 2.74 -> 5.86 |
| ATT [8555] F/F | 2.2% -> 6.5% -> 85.1% | 2.95 -> 7.23 |
| ATT [8555] T/T | 2.6% -> 14.0% -> 98.6% | 2.08 -> 6.87 |
| ATT [85555] F/F | 0.4% -> 0.9% -> 42.7% | 2.54 -> 8.23 |
| MLP [8555] F/F | 42.7% -> 87.9% -> 84.2% | 6.37 -> 6.71 |

Successful runs *converge* toward a small code set. The failing
attention runs start almost fully collapsed (a few dozen codes) and then
*expand* to fill the codebook as training proceeds, which is the
signature of the quantizer losing its grip on the latent rather than
the latent being over-compressed.

A second view is how entropy changes *within* one inference, from
recursion step 1 to step 16, at the final checkpoint:

| Run | Step 1 | Step 8 | Step 16 | Change |
|---|---|---|---|---|
| ATT res0.3 F/F | 3.76 | 1.89 | 1.64 | -2.12 |
| MLP res0.3 F/F | 4.56 | 2.88 | 2.88 | -1.68 |
| ATT EM(1.0) F/F | 7.56 | 6.04 | 5.86 | -1.70 |
| MLP [85555] res0.5 T/T | 8.01 | 6.67 | 6.55 | -1.46 |
| Maze ATT T/T | 2.19 | 2.07 | 2.08 | -0.11 |
| ATT [8555] F/F | 7.40 | 7.24 | 7.23 | -0.16 |
| ATT [85555] F/F | 8.38 | 8.23 | 8.23 | -0.16 |

Models that solve the task sharpen their discrete state as the
recursion proceeds; models that fail stay diffuse. Maze is already sharp
at step 1, which is consistent with it being the easier task for this
architecture.

## 5. What changed in the code as a result

1. **Residual mixing** (`fsq_residual_mode`, `fsq_residual_weight`).
   Blending the quantized latent with the continuous one
   (`alpha = 0.3`) recovers TRM-Att Sudoku from 0.32 to 0.77 and gives
   the best TRM-MLP result, 0.84. The interpretation we find most
   convincing is a rank argument: the quantized path
   `project_out(code)` has rank at most `len(fsq_levels)` (four here),
   so a pure quantization bottleneck squeezes a 512-dimensional state
   into a rank-4 subspace between recursion steps. The residual keeps a
   high-rank continuous path and lets the discrete path act as a
   correction. `learned_scalar` mode parametrises `alpha` through a
   sigmoid and learns it.
2. **Pre-tanh normalisation** (`fsq_pre_norm`). Standardising the
   projected latent per codebook dimension before `tanh` removes the
   saturation that produces the U-shaped level histograms. On Maze with
   `alpha = 0.3` and `fsq_recon_weight = 0.1` it flattens the per-dimension
   histograms (dimension entropies rise from about 1.9 / 1.5 / 0.7 / 0.6
   bits to about 2.8 / 3.0 / 2.2 / 2.3) and keeps utilization around 9 to
   12 percent, while cosine similarity between latent and code drops to
   0.77 to 0.92. Combined with the residual, the reconstruction loss and
   exact-match up-weighting it gives the best TRM-Att Sudoku result
   (0.8303). On Maze it needs sampling on (`T/T`); trained greedily the
   normalized codes drift and accuracy collapses to 0.43. It therefore
   stays off by default and is enabled per recipe.
3. **Reconstruction loss** (`fsq_recon_weight` in TRM, `lambda_recon` in
   QRM). A small MSE between the quantized and the detached continuous
   latent keeps the two paths aligned when they are blended.

Two follow-ups were designed but not implemented: binary levels
(`levels=[2]*k`) to make the effective binarisation explicit, and an
SVD-based effective-rank tracker for `z_L` across recursion steps. Both
are natural extensions of `FSQEntropyAnalyzer`.

## 6. Reproducing

```bash
# One checkpoint, three recursion steps, greedy quantization
uv run python -m qrm.analyze \
    --ckpt_dir results/<run>/final_checkpoint \
    --output_dir analysis/fsq_entropy/<name> \
    --analyzer fsq_entropy \
    --analyze_rounds 1,8,16 \
    --fsq_sampling_inference false \
    --batch_size 768 --max_samples 1024

# Compare several checkpoints
uv run python scripts/analysis/fsq_entropy_report.py \
    --manifest scripts/analysis/fsq_entropy_manifest.example.json \
    --output analysis/fsq_entropy/comparison_report.md
```

`scripts/analysis/fsq_entropy_report.py` takes a JSON manifest that maps
a label to a summary file and, optionally, an accuracy, and emits the
comparison and trajectory tables shown above.
