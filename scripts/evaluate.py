#!/usr/bin/env python3
"""
Evaluation script for Qwen3-4B using lm-eval.

Usage:
    python scripts/evaluate.py --model outputs/sft/final --tasks mmlu,hellaswag
    python scripts/evaluate.py --model outputs/dpo/final --all  # Run all benchmarks
"""

import argparse
import os
import subprocess
import json
from datetime import datetime


# All benchmarks from the spec
ALL_TASKS = [
    "mmlu",
    "hellaswag", 
    "arc_challenge",
    "winogrande",
    "piqa",
    "gsm8k",
    "humaneval",
    "ifeval",
    "commonsense_qa",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,mmlu",
        help="Comma-separated list of tasks to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for quick testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    return parser.parse_args()


def setup_environment():
    """Setup environment variables for offline evaluation."""
    # Set cache path for benchmarks
    os.environ["HF_HOME"] = "/data/share/benchmark_cache"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("Environment configured for offline evaluation")


def run_lm_eval(
    model_path: str,
    tasks: list,
    batch_size: int = 8,
    limit: int = None,
    output_path: str = None,
    device: str = "cuda",
) -> dict:
    """
    Run lm-eval harness.
    
    Args:
        model_path: Path to model
        tasks: List of task names
        batch_size: Evaluation batch size
        limit: Sample limit per task
        output_path: Path to save results
        device: Device to use
        
    Returns:
        Evaluation results dictionary
    """
    tasks_str = ",".join(tasks)
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks_str,
        "--batch_size", str(batch_size),
        "--device", device,
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if output_path:
        cmd.extend(["--output_path", output_path])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode


def quick_eval(model_path: str, batch_size: int = 4) -> None:
    """Run quick evaluation on a subset of tasks."""
    quick_tasks = ["hellaswag", "piqa"]  # Fast to evaluate
    
    print("=" * 50)
    print("Quick Evaluation (subset of tasks)")
    print("=" * 50)
    
    run_lm_eval(
        model_path=model_path,
        tasks=quick_tasks,
        batch_size=batch_size,
        limit=100,  # Only 100 samples per task
    )


def full_eval(model_path: str, output_dir: str, batch_size: int = 8) -> None:
    """Run full evaluation on all benchmarks."""
    print("=" * 50)
    print("Full Evaluation (all benchmarks)")
    print("=" * 50)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path)
    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Run evaluation
    run_lm_eval(
        model_path=model_path,
        tasks=ALL_TASKS,
        batch_size=batch_size,
        output_path=output_path,
    )
    
    print(f"\nResults saved to: {output_path}")


def main():
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine tasks to run
    if args.all:
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
    
    print(f"Model: {args.model}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    
    # Run evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"eval_{timestamp}")
    
    return_code = run_lm_eval(
        model_path=args.model,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=output_path,
        device=args.device,
    )
    
    if return_code == 0:
        print(f"\n✓ Evaluation complete! Results saved to: {output_path}")
    else:
        print(f"\n✗ Evaluation failed with return code: {return_code}")
    
    return return_code


if __name__ == "__main__":
    exit(main())
