"""Entry point for inference analysis."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run inference analysis on checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: ckpt_dir/analysis)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze (default: all)")
    parser.add_argument("--batch_size", type=int, default=768, help="Batch size for inference (default: 768)")
    parser.add_argument("--analyzer", type=str, default="trm", choices=["trm", "fsq_entropy"], help="Analyzer type")
    parser.add_argument("--fsq_sampling_inference", type=str, default=None, choices=["true", "false"],
                        help="Override fsq_sampling_inference from checkpoint config (default: use checkpoint config)")
    parser.add_argument("--save_preds", action="store_true",
                        help="Save per-sample predictions (increases output size significantly)")
    parser.add_argument("--analyze_rounds", type=str, default="1,8,16",
                        help="Comma-separated round numbers to analyze (1-indexed, for fsq_entropy analyzer)")
    args = parser.parse_args()

    # Parse fsq_sampling_inference override
    fsq_sampling_inference = None
    if args.fsq_sampling_inference is not None:
        fsq_sampling_inference = args.fsq_sampling_inference == "true"

    output_dir = args.output_dir or f"{args.ckpt_dir}/analysis"

    if args.analyzer == "trm":
        from qrm.analyzers.trm_analyzer import TRMAnalyzer
        analyzer = TRMAnalyzer(
            ckpt_dir=args.ckpt_dir,
            output_dir=output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            fsq_sampling_inference=fsq_sampling_inference,
            save_preds=args.save_preds,
        )
    elif args.analyzer == "fsq_entropy":
        from qrm.analyzers.fsq_entropy_analyzer import FSQEntropyAnalyzer
        analyze_rounds = tuple(int(r) - 1 for r in args.analyze_rounds.split(","))
        analyzer = FSQEntropyAnalyzer(
            ckpt_dir=args.ckpt_dir,
            output_dir=output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            fsq_sampling_inference=fsq_sampling_inference,
            analyze_rounds=analyze_rounds,
        )

    analyzer.run()


if __name__ == "__main__":
    main()
