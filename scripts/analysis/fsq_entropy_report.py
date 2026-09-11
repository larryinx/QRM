"""Build markdown reports from ``fsq_entropy_summary.json`` files.

The FSQ entropy analyzer (``python -m qrm.analyze --analyzer fsq_entropy``)
writes one summary per checkpoint. This script compares several of them.
It is driven by a JSON manifest so that no run names or paths are
hard-coded:

    {
      "round": "round_16",
      "experiments": {
        "<label>": {"summary": "<path/to/fsq_entropy_summary.json>",
                    "exact_acc": 0.7694}          # exact_acc is optional
      },
      "trajectories": {                            # optional
        "<label>": {"Early": "<summary.json>", "Mid": "<summary.json>",
                    "Final": "<summary.json>"}
      }
    }

Usage:
    python scripts/analysis/fsq_entropy_report.py --manifest manifest.json \
        [--output report.md]

``experiments`` produces the per-step comparison tables (D1-D5 and the
residual configuration) for every step present in the summaries.
``trajectories`` produces training-trajectory tables from the final step
of each listed checkpoint, plus the within-inference progression of the
position entropy. Missing files are reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"WARNING: missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([head, sep] + body)


def last_round(summary: dict, preferred: Optional[str] = None) -> Optional[str]:
    rounds = sorted(summary.get("per_round", {}).keys())
    if not rounds:
        return None
    if preferred in rounds:
        return preferred
    return rounds[-1]


def fmt_acc(meta: dict) -> str:
    acc = meta.get("exact_acc")
    return f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"


# ----------------------------------------------------------------------------
# per-step comparison
# ----------------------------------------------------------------------------


def comparison_section(experiments: Dict[str, dict], round_key: str) -> str:
    lines = [f"## Step {round_key.replace('round_', '')} comparison\n"]

    # D1
    rows = []
    for label, exp in experiments.items():
        rd = exp["data"]["per_round"].get(round_key)
        if rd is None:
            continue
        d1 = rd["d1_codebook_utilization"]
        rows.append([
            label, fmt_acc(exp),
            f"{d1['active_codes']}/{d1['codebook_size']}",
            f"{d1['utilization_rate']:.1%}",
            f"{d1['top10_codes_cover_pct']:.1%}",
            f"{d1['top50_codes_cover_pct']:.1%}",
            d1["verdict"],
        ])
    lines += ["### D1: codebook utilization\n",
              md_table(["Experiment", "exact_acc", "Active codes", "Utilization",
                        "Top-10 cover", "Top-50 cover", "Verdict"], rows), ""]

    # D2
    rows = []
    for label, exp in experiments.items():
        rd = exp["data"]["per_round"].get(round_key)
        if rd is None:
            continue
        d2 = rd["d2_position_entropy"]
        rows.append([
            label, fmt_acc(exp),
            f"{d2['mean_entropy']:.2f}",
            f"{d2['answer_positions_mean']:.2f}",
            f"{d2['prompt_positions_mean']:.2f}",
            f"{d2['max_possible_entropy']:.2f}",
            d2["verdict"],
        ])
    lines += ["### D2: per-position code entropy (bits)\n",
              md_table(["Experiment", "exact_acc", "Mean", "Answer positions",
                        "Prompt positions", "Max possible", "Verdict"], rows), ""]

    # D3
    rows = []
    for label, exp in experiments.items():
        rd = exp["data"]["per_round"].get(round_key)
        if rd is None:
            continue
        d3 = rd["d3_topk_gap"]
        rows.append([
            label, fmt_acc(exp),
            f"{d3['mean_top1_prob']:.6f}",
            f"{d3['mean_gap_1_2']:.4f}",
            f"{d3['mean_gap_1_k']:.4f}",
        ])
    lines += ["### D3: top-k probability gap\n",
              md_table(["Experiment", "exact_acc", "Top-1 prob", "Gap 1-2", "Gap 1-k"], rows), ""]

    # D4 (variable number of dims)
    max_dims = 0
    for exp in experiments.values():
        rd = exp["data"]["per_round"].get(round_key)
        if rd is not None:
            max_dims = max(max_dims, len(rd["d4_per_dim_levels"]["dims"]))
    headers = ["Experiment", "exact_acc"] + [f"Dim{i}" for i in range(max_dims)] + ["Eff. codebook"]
    rows = []
    for label, exp in experiments.items():
        rd = exp["data"]["per_round"].get(round_key)
        if rd is None:
            continue
        d4 = rd["d4_per_dim_levels"]
        cells = [
            f"{d['entropy']:.2f}/{d['max_entropy']:.2f} ({d['active_levels']}/{d['levels']})"
            for d in d4["dims"]
        ]
        cells += ["-"] * (max_dims - len(cells))
        rows.append([label, fmt_acc(exp)] + cells + [str(d4["effective_codebook_size"])])
    lines += ["### D4: per-dimension level entropy (entropy/max, active/levels)\n",
              md_table(headers, rows), ""]

    # D5
    rows = []
    for label, exp in experiments.items():
        rd = exp["data"]["per_round"].get(round_key)
        if rd is None:
            continue
        d5 = rd["d5_quantization_error"]
        rows.append([label, fmt_acc(exp), f"{d5['mean_mse']:.6f}",
                     f"{d5['mean_cosine_similarity']:.4f}"])
    lines += ["### D5: quantization error\n",
              md_table(["Experiment", "exact_acc", "MSE", "Cosine sim"], rows), ""]

    # residual configuration
    rows = []
    for label, exp in experiments.items():
        cfg = exp["data"].get("config", {})
        res = cfg.get("fsq_residual", {})
        rows.append([label, str(cfg.get("fsq_levels", "-")), str(res.get("mode", "-")),
                     str(res.get("alpha", "-"))])
    lines += ["### FSQ configuration\n",
              md_table(["Experiment", "Levels", "Residual mode", "Alpha"], rows), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# training trajectories
# ----------------------------------------------------------------------------


def trajectory_section(trajectories: Dict[str, Dict[str, dict]], preferred_round: str) -> str:
    lines = ["## Training trajectories\n",
             "Diagnostics at the final analysed recursion step of each checkpoint.\n"]

    for label, ckpts in trajectories.items():
        lines.append(f"### {label}\n")
        rows = []
        dim_rows = []
        dim_headers = None
        for ck_label, data in ckpts.items():
            rk = last_round(data, preferred_round)
            if rk is None:
                continue
            rd = data["per_round"][rk]
            d1, d2, d5 = (rd["d1_codebook_utilization"], rd["d2_position_entropy"],
                          rd["d5_quantization_error"])
            rows.append([ck_label, f"{d1['active_codes']}/{d1['codebook_size']}",
                         f"{d1['utilization_rate']:.1%}",
                         f"{d2['mean_entropy']:.2f}/{d2['max_possible_entropy']:.2f}",
                         f"{d5['mean_mse']:.6f}", f"{d5['mean_cosine_similarity']:.4f}"])
            dims = rd["d4_per_dim_levels"]["dims"]
            if dim_headers is None:
                dim_headers = ["Checkpoint"] + [f"Dim{d['dim']} ({d['levels']})" for d in dims]
            dim_rows.append([ck_label] + [f"{d['entropy']:.2f}/{d['max_entropy']:.2f}" for d in dims])
        lines += [md_table(["Checkpoint", "Active codes", "Utilization", "Mean entropy",
                            "MSE", "Cosine sim"], rows), ""]
        if dim_headers:
            lines += [md_table(dim_headers, dim_rows), ""]

    # cross-experiment utilization / entropy trends
    rows_u, rows_e = [], []
    for label, ckpts in trajectories.items():
        vals_u, vals_e, max_e = [], [], None
        for _, data in ckpts.items():
            rk = last_round(data, preferred_round)
            if rk is None:
                continue
            rd = data["per_round"][rk]
            vals_u.append(rd["d1_codebook_utilization"]["utilization_rate"])
            vals_e.append(rd["d2_position_entropy"]["mean_entropy"])
            max_e = rd["d2_position_entropy"]["max_possible_entropy"]
        if len(vals_u) >= 2:
            rows_u.append([label] + [f"{v:.1%}" for v in vals_u]
                          + [f"{(vals_u[-1] - vals_u[0]) * 100:+.1f} pp"])
            rows_e.append([label] + [f"{v:.2f}" for v in vals_e]
                          + [f"{max_e:.2f}", f"{vals_e[-1] - vals_e[0]:+.2f}"])
    if rows_u:
        n = max(len(r) for r in rows_u) - 2
        lines += ["### Codebook utilization across checkpoints\n",
                  md_table(["Experiment"] + [f"ckpt {i + 1}" for i in range(n)] + ["Change"], rows_u), ""]
        lines += ["### Position entropy across checkpoints\n",
                  md_table(["Experiment"] + [f"ckpt {i + 1}" for i in range(n)] + ["Max", "Change"], rows_e), ""]

    # within-inference progression
    lines += ["### Within-inference progression of position entropy\n"]
    for label, ckpts in trajectories.items():
        rows = []
        round_keys = None
        for ck_label, data in ckpts.items():
            pr = data["per_round"]
            keys = sorted(pr.keys())
            if not keys:
                continue
            round_keys = keys
            ents = [pr[k]["d2_position_entropy"]["mean_entropy"] for k in keys]
            rows.append([ck_label] + [f"{e:.2f}" for e in ents] + [f"{ents[-1] - ents[0]:+.2f}"])
        if rows:
            lines += [f"**{label}**\n",
                      md_table(["Checkpoint"] + [k.replace("round_", "step ") for k in round_keys]
                               + ["First to last"], rows), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSON manifest (see module docstring)")
    parser.add_argument("--output", default=None, help="Write the markdown report here (default: stdout only)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    base = os.path.dirname(os.path.abspath(args.manifest))

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.getcwd(), p)) \
            if os.path.exists(p) else os.path.normpath(os.path.join(base, p))

    preferred_round = manifest.get("round", "round_16")
    parts = ["# FSQ entropy report\n"]

    experiments: Dict[str, dict] = {}
    for label, meta in manifest.get("experiments", {}).items():
        data = load_json(resolve(meta["summary"]))
        if data is not None:
            experiments[label] = {**meta, "data": data}
    if experiments:
        rounds = set()
        for exp in experiments.values():
            rounds.update(exp["data"].get("per_round", {}).keys())
        for rk in sorted(rounds):
            parts.append(comparison_section(experiments, rk))

    trajectories: Dict[str, Dict[str, dict]] = {}
    for label, ckpts in manifest.get("trajectories", {}).items():
        loaded = {}
        for ck_label, path in ckpts.items():
            data = load_json(resolve(path))
            if data is not None:
                loaded[ck_label] = data
        if loaded:
            trajectories[label] = loaded
    if trajectories:
        parts.append(trajectory_section(trajectories, preferred_round))

    if not experiments and not trajectories:
        print("Nothing to report: no summary files could be loaded.", file=sys.stderr)
        sys.exit(1)

    report = "\n".join(parts)
    print(report)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
