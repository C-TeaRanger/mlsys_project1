#!/usr/bin/env python3
"""
Data preparation script for Qwen3-4B fine-tuning.

Usage:
    python scripts/prepare_data.py --dataset alpaca --output_dir data/processed
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.data.dataset import load_sft_dataset, load_preference_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt", "openorca", "ultrafeedback"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/share/Qwen3-4B-Base",
        help="Path to model for tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.02,
        help="Validation split size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    # Prepare dataset based on type
    if args.dataset in ["alpaca", "sharegpt", "openorca"]:
        print(f"Preparing SFT dataset: {args.dataset}")
        dataset = load_sft_dataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            validation_size=args.validation_size,
            seed=args.seed,
        )
        
        # Save processed data
        output_path = os.path.join(args.output_dir, f"{args.dataset}_sft")
        print(f"Saving to: {output_path}")
        dataset.save_to_disk(output_path)
        
        # Print statistics
        print("\n=== Dataset Statistics ===")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        
        # Sample length distribution
        train_lengths = [len(x["input_ids"]) for x in dataset["train"]]
        print(f"Sequence length - Min: {min(train_lengths)}, Max: {max(train_lengths)}, Avg: {sum(train_lengths)/len(train_lengths):.1f}")
        
    elif args.dataset in ["ultrafeedback"]:
        print(f"Preparing preference dataset: {args.dataset}")
        dataset = load_preference_dataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )
        
        # Save processed data
        output_path = os.path.join(args.output_dir, f"{args.dataset}_preference")
        print(f"Saving to: {output_path}")
        dataset.save_to_disk(output_path)
        
        print(f"\nTotal samples: {len(dataset)}")
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
