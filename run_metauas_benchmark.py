#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import benchmark_anomaly_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MetaUAS on a defect folder")
    parser.add_argument(
        "--defect-dir",
        "--data-dir",
        dest="defect_dir",
        default="/home/awais/Datasets/benchmark_images",
        help="Folder of defect images",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Folder of normal images for prompt selection",
    )
    parser.add_argument(
        "--good-images-dir",
        default=None,
        help="Folder of good images for CLIP similarity prompt selection",
    )
    parser.add_argument(
        "--use-sorted-good-images",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use sorted good images for nearest-neighbor prompts",
    )
    parser.add_argument("--async-preprocess", action="store_true", help="Preprocess in a background worker")
    parser.add_argument("--prefetch-size", type=int, default=32, help="Async preprocess queue size")
    parser.add_argument("--async-save", action="store_true", help="Save outputs in a background worker")
    parser.add_argument("--save-queue-size", type=int, default=300, help="Async save queue size")
    parser.add_argument("--prompt-path", default=None, help="Single prompt image path (overrides reference-dir)")
    parser.add_argument("--checkpoint", default="weights/metauas-512.ckpt", help="MetaUAS checkpoint path")
    parser.add_argument("--image-size", type=int, default=512, help="Input size for MetaUAS")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--prompt-mode", default="first", choices=["first", "random"], help="Prompt selection strategy")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for throughput benchmark")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (count)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    parser.add_argument(
        "--results-dir",
        default="/home/awais/Datasets/gm_anomaly_test_results/metauas",
        help="Results root",
    )
    parser.add_argument("--enable-embeddings", action="store_true", help="Enable embedding/index benchmarks")
    parser.add_argument("--update-count", type=int, default=0, help="Number of queries to measure index update cost")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    defect_dir = Path(args.defect_dir)
    if not defect_dir.exists():
        raise FileNotFoundError(f"defect-dir not found: {defect_dir}")

    if args.reference_dir is None or args.good_images_dir is None:
        good_root = (
            "/home/awais/Datasets/gm_good_images_sorted"
            if args.use_sorted_good_images
            else "/home/awais/Datasets/gm_good_images"
        )
        if args.reference_dir is None:
            args.reference_dir = good_root
        if args.good_images_dir is None:
            args.good_images_dir = good_root

    if args.prompt_path:
        prompt_path = Path(args.prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt-path not found: {prompt_path}")
    elif args.reference_dir or args.good_images_dir:
        if args.reference_dir:
            ref_dir = Path(args.reference_dir)
            if not ref_dir.exists():
                raise FileNotFoundError(f"reference-dir not found: {ref_dir}")
        if args.good_images_dir:
            good_dir = Path(args.good_images_dir)
            if not good_dir.exists():
                raise FileNotFoundError(f"good-images-dir not found: {good_dir}")
    else:
        raise ValueError("Provide --prompt-path or --reference-dir for one-shot prompting")

    model_kwargs = {
        "checkpoint": args.checkpoint,
        "image_size": args.image_size,
        "device": args.device,
        "prompt_path": args.prompt_path,
        "reference_dir": args.reference_dir,
        "prompt_mode": args.prompt_mode,
        "good_images_dir": args.good_images_dir,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    argv = [
        "benchmark_anomaly_models.py",
        "--model-module",
        "metauas_benchmark_adapter",
        "--model-class",
        "MetaUASAdapter",
        "--model-kwargs",
        json.dumps(model_kwargs),
        "--data-dir",
        str(defect_dir),
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--results-dir",
        args.results_dir,
    ]

    if args.limit and args.limit > 0:
        argv += ["--limit", str(args.limit)]
    if args.reference_dir:
        argv += ["--reference-dir", args.reference_dir]
    if args.enable_embeddings:
        argv.append("--enable-embeddings")
    if args.update_count:
        argv += ["--update-count", str(args.update_count)]
    if args.async_preprocess:
        argv.append("--async-preprocess")
        argv += ["--prefetch-size", str(args.prefetch_size)]
    if args.async_save:
        argv.append("--async-save")
        argv += ["--save-queue-size", str(args.save_queue_size)]

    sys.argv = argv
    benchmark_anomaly_models.main()


if __name__ == "__main__":
    main()
