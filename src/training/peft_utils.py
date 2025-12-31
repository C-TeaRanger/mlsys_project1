"""PEFT (LoRA/QLoRA) utilities for model fine-tuning."""

from typing import Optional, List, Dict, Any
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
)


def create_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    Create LoRA configuration for fine-tuning.
    
    Args:
        r: LoRA rank (higher = more parameters, better quality)
        lora_alpha: LoRA alpha (scaling factor, typically 2*r)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias training mode ("none", "all", "lora_only")
        task_type: Task type for PEFT
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default target modules for Qwen models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )


def create_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes quantization config for QLoRA.
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_use_double_quant: Whether to use nested quantization
        
    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def load_model_for_training(
    model_name_or_path: str,
    use_lora: bool = True,
    load_in_4bit: bool = True,
    lora_config: Optional[LoraConfig] = None,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    device_map: str = "auto",
    use_gradient_checkpointing: bool = True,  # 新增参数
) -> PreTrainedModel:
    """
    Load model with optional LoRA and quantization for training.
    
    Args:
        model_name_or_path: Model name or path
        use_lora: Whether to use LoRA
        load_in_4bit: Whether to use 4-bit quantization
        lora_config: LoRA configuration (creates default if None)
        torch_dtype: Model dtype
        attn_implementation: Attention implementation
        device_map: Device mapping strategy
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        Model ready for training
    """
    import os
    
    # 检测是否在分布式训练环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    
    # 对于分布式训练 + 4-bit 量化，需要使用特定的 device_map
    if is_distributed and load_in_4bit:
        # 每个进程只加载到当前 GPU
        device_map = {"": local_rank}
        print(f"Distributed training detected (rank={local_rank}), using device_map={device_map}")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": getattr(torch, torch_dtype),
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    # Set attention implementation
    model_kwargs["attn_implementation"] = attn_implementation
    
    # Add quantization config if needed
    if load_in_4bit:
        model_kwargs["quantization_config"] = create_quantization_config()
    
    # Load base model
    print(f"Loading model from: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    
    # Prepare for k-bit training
    if load_in_4bit:
        print(f"Preparing model for k-bit training (gradient_checkpointing={use_gradient_checkpointing})")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    
    # Apply LoRA
    if use_lora:
        if lora_config is None:
            lora_config = create_lora_config()
        
        print(f"Applying LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def prepare_model_for_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    use_gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """
    Prepare an already-loaded model for LoRA training.
    
    Args:
        model: Pre-loaded model
        lora_config: LoRA configuration
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        Model with LoRA applied
    """
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def merge_and_save_lora(
    model: PreTrainedModel,
    output_path: str,
    tokenizer = None,
) -> None:
    """
    Merge LoRA weights into base model and save.
    
    Args:
        model: Model with LoRA adapters
        output_path: Path to save merged model
        tokenizer: Optional tokenizer to save alongside
    """
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)
    
    print("Done!")
