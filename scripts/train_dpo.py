#!/usr/bin/env python3
"""
DPO Training script for Qwen3-4B.

Usage:
    # Single GPU
    python scripts/train_dpo.py --config configs/dpo_config.yaml
    
    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=4 scripts/train_dpo.py --config configs/dpo_config.yaml
"""

import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from transformers import AutoTokenizer
from src.training.dpo_trainer import DPOTrainerWrapper, DPOTrainingConfig
from src.data.dataset import load_preference_dataset
from src.utils.logging import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
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
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    
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
    """Filter config to only include fields supported by DPOTrainingConfig."""
    supported_fields = {
        # Model
        "policy_model", "reference_model", "torch_dtype", "attn_implementation",
        # LoRA
        "use_lora", "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        # Quantization
        "load_in_4bit",
        # DPO specific
        "beta", "loss_type",
        # Training
        "output_dir", "learning_rate", "lr_scheduler_type", "warmup_ratio",
        "weight_decay", "max_grad_norm",
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "num_train_epochs", "max_steps",
        "eval_steps", "save_steps", "logging_steps",
        "gradient_checkpointing", "bf16",
        "seed",
        # Data
        "max_seq_length", "max_prompt_length",
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
    if args.beta:
        filtered_config["beta"] = args.beta
    if args.learning_rate:
        filtered_config["learning_rate"] = args.learning_rate
    if args.max_steps:
        filtered_config["max_steps"] = args.max_steps
    
    # Create training config
    training_config = DPOTrainingConfig(**filtered_config)
    
    # Load tokenizer for data loading
    logger.info(f"Loading tokenizer from: {training_config.policy_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        training_config.policy_model,
        trust_remote_code=True,
    )
    
    # Load or prepare preference dataset
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset_name", "ultrafeedback")
    processed_path = f"data/processed/{dataset_name}_preference"
    
    if os.path.exists(processed_path):
        logger.info(f"Loading processed data from: {processed_path}")
        dataset = load_from_disk(processed_path)
    else:
        logger.info(f"Loading preference dataset: {dataset_name}")
        dataset = load_preference_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=training_config.max_seq_length,
        )
    
    logger.info(f"Total preference samples: {len(dataset)}")
    
    # Split for validation if needed
    if isinstance(dataset, dict) and "train" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("validation", None)
    else:
        # Split dataset
        split = dataset.train_test_split(test_size=0.02, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    
    # Create trainer and run
    trainer_wrapper = DPOTrainerWrapper(training_config)
    trainer_wrapper.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    logger.info("DPO Training complete!")


if __name__ == "__main__":
    main()
