# Documentation

| Document | Contents |
|---|---|
| [01_overview.md](01_overview.md) | The idea: TRM, the quantized hand-off (FSQ), stochastic top-k quantization, tree search, the training objective, and how the code differs from the paper |
| [02_codebase.md](02_codebase.md) | Package layout, data flow, model classes, losses, search, trainers, config keys, entry points, engineering notes and known limitations |
| [03_training_guide.md](03_training_guide.md) | Environment with uv, data preparation, launching TRM / TRM + FSQ / QRM on each task, multi-GPU, resuming, evaluation, tests |
| [04_datasets.md](04_datasets.md) | Sudoku-Extreme, Maze-Hard, ARC-AGI-1/2 builders, variants and identifier modes |
| [05_analysis_tools.md](05_analysis_tools.md) | The `qrm.analyze` diagnostics, the summary JSON schema and the report script |
| [06_fsq_entropy_analysis.md](06_fsq_entropy_analysis.md) | The codebook study: what "entropy collapse" turned out to mean, and the design changes it caused |
| [07_experiments.md](07_experiments.md) | Replication numbers, TRM + FSQ ablations, QRM results, unfinished runs |
| [08_design_notes.md](08_design_notes.md) | Reasoning behind the non-obvious decisions and what to do next |
