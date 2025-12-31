#!/usr/bin/env python3
"""
Merge LoRA weights into base model.

Usage:
    python scripts/merge_lora.py --model outputs/sft/final --output outputs/sft/merged
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model with LoRA adapters",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/data/share/Qwen3-4B-Base",
        help="Path to base model (if LoRA was trained separately)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="Model dtype",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.torch_dtype),
        trust_remote_code=True,
        device_map="auto",
    )
    
    print(f"Loading LoRA adapters from: {args.model}")
    model = PeftModel.from_pretrained(base_model, args.model)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    merged_model.save_pretrained(args.output)
    
    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(args.output)
    
    print("✓ Merge complete!")
    print(f"  Merged model saved to: {args.output}")


if __name__ == "__main__":
    main()
