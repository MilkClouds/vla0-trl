#!/usr/bin/env python3
"""VLA-0 Evaluation Script for LIBERO benchmark."""

import argparse
from datetime import datetime
from pathlib import Path

from rv_eval.evaluator import LiberoEvaluator
from rv_train.model import QwenVLActor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLA-0 on LIBERO")

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to dataset_stats.json (auto-detected if not specified)"
    )

    # Evaluation settings
    parser.add_argument(
        "--task_suite",
        type=str,
        default=None,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="Task suite to evaluate (all if not specified)",
    )
    parser.add_argument("--task_name", type=str, default=None, help="Specific task to evaluate")

    # Hyperparameters (match paper defaults)
    parser.add_argument("--action_horizon", type=int, default=1)
    parser.add_argument("--frame_skip", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ensemble_prediction", type=int, default=1, help="Number of action chunks to ensemble")
    parser.add_argument(
        "--ensemble_version",
        type=int,
        default=1,
        choices=[1, 2],
        help="Ensemble version: 1=flat weight, 2=exponential decay",
    )
    parser.add_argument(
        "--ensemble_weight", type=float, default=0.5, help="Weight for ensemble (0.5 for both versions)"
    )

    # Sharding for parallel evaluation
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard index (0 to num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards to split evaluation")

    # Model params
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--num_bins", type=int, default=1000)
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile for model")

    # Image processing (match training)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.875)
    parser.add_argument("--tile_images", action="store_true", default=True)

    # Output
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Log directory (auto-generated from model path if not specified)"
    )
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="save_video", action="store_false")
    parser.add_argument("--skip_evaluated", action="store_true", help="Skip already evaluated episodes")

    return parser.parse_args()


def build_log_dir(args, timestamp: str) -> str:
    """Build log directory path: eval_logs/{model_name}/{timestamp}"""
    model_name = (
        Path(Path(args.model_path).parent.name) / Path(args.model_path).name
        if "checkpoint-" in args.model_path
        else Path(args.model_path).name
    )
    return str(Path("eval_logs") / model_name / timestamp)


def main():
    args = parse_args()

    # Auto-detect stats path
    stats_path = args.stats_path
    if stats_path is None:
        model_dir = Path(args.model_path).parent
        candidate = model_dir / "dataset_stats.json"
        if candidate.exists():
            stats_path = str(candidate)
        else:
            # Try parent directory
            candidate = model_dir.parent / "dataset_stats.json"
            if candidate.exists():
                stats_path = str(candidate)

    if stats_path is None:
        raise ValueError("Could not find dataset_stats.json. Specify --stats_path")

    # Build log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or build_log_dir(args, timestamp)

    print(f"Loading model from: {args.model_path}")
    print(f"Loading stats from: {stats_path}")
    print(f"Logs will be saved to: {log_dir}")

    model = QwenVLActor(
        model_path=args.model_path,
        stats_path=stats_path,
        horizon=args.horizon,
        action_dim=args.action_dim,
        num_bins=args.num_bins,
        torch_compile=args.torch_compile,
    )

    evaluator = LiberoEvaluator(
        model=model,
        log_dir=log_dir,
        save_video=args.save_video,
        seed=args.seed,
        action_horizon=args.action_horizon,
        frame_skip=args.frame_skip,
        img_size=args.img_size,
        crop_ratio=args.crop_ratio,
        tile_images=args.tile_images,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        skip_evaluated=args.skip_evaluated,
        ensemble_prediction=args.ensemble_prediction,
        ensemble_version=args.ensemble_version,
        ensemble_weight=args.ensemble_weight,
    )

    if args.num_shards > 1:
        print(f"Shard {args.shard_id}/{args.num_shards}")
    print("Starting evaluation...")
    evaluator.evaluate(
        task_suite_name=args.task_suite,
        task_name=args.task_name,
    )

    print(f"\nResults saved to: {log_dir}/")


if __name__ == "__main__":
    main()
