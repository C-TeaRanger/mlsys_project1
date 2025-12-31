#!/usr/bin/env python3
"""
SFT Training script for Qwen3-4B.

Usage:
    # Single GPU
    python scripts/train_sft.py --config configs/sft_config.yaml
    
    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=4 scripts/train_sft.py --config configs/sft_config.yaml
"""

import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from transformers import AutoTokenizer
from src.training.sft_trainer import SFTTrainerWrapper, SFTTrainingConfig
from src.data.dataset import load_sft_dataset
from src.utils.logging import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    # Allow overriding config values
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def flatten_config(config: dict) -> dict:
    """Flatten nested config dictionary."""
    flat = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    return flat


def filter_config_for_dataclass(flat_config: dict) -> dict:
    """Filter config to only include fields supported by SFTTrainingConfig."""
    # SFTTrainingConfig 支持的字段
    supported_fields = {
        # Model
        "base_model", "torch_dtype", "attn_implementation",
        # LoRA
        "use_lora", "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        # Quantization
        "load_in_4bit",
        # Training
        "output_dir", "learning_rate", "lr_scheduler_type", "warmup_ratio",
        "weight_decay", "max_grad_norm",
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "num_train_epochs", "max_steps",
        "eval_steps", "save_steps", "logging_steps",
        "gradient_checkpointing", "bf16",
        "seed",
        # Data
        "max_seq_length",
    }
    
    filtered = {k: v for k, v in flat_config.items() if k in supported_fields}
    return filtered


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    flat_config = flatten_config(config)
    
    # Filter to only supported fields
    filtered_config = filter_config_for_dataclass(flat_config)
    
    # Override with CLI args
    if args.output_dir:
        filtered_config["output_dir"] = args.output_dir
    if args.learning_rate:
        filtered_config["learning_rate"] = args.learning_rate
    if args.max_steps:
        filtered_config["max_steps"] = args.max_steps
    if args.lora_r:
        filtered_config["lora_r"] = args.lora_r
    
    # Create training config
    training_config = SFTTrainingConfig(**filtered_config)
    
    # Load tokenizer for data loading
    logger.info(f"Loading tokenizer from: {training_config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        training_config.base_model,
        trust_remote_code=True,
    )
    
    # Load or prepare dataset
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset_name", "alpaca")
    processed_path = f"data/processed/{dataset_name}_sft"
    
    if os.path.exists(processed_path):
        logger.info(f"Loading processed data from: {processed_path}")
        dataset = load_from_disk(processed_path)
    else:
        logger.info(f"Loading and processing dataset: {dataset_name}")
        dataset = load_sft_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=training_config.max_seq_length,
            validation_size=data_config.get("validation_size", 0.02),
        )
    
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")
    
    # Create trainer and run
    trainer_wrapper = SFTTrainerWrapper(training_config)
    trainer_wrapper.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
