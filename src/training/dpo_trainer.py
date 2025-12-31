"""DPO Trainer wrapper for Qwen3-4B fine-tuning."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import os

import torch
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from datasets import Dataset

from .peft_utils import create_lora_config, load_model_for_training


@dataclass  
class DPOTrainingConfig:
    """Configuration for DPO training."""
    
    # Model
    policy_model: str = "outputs/sft/final"
    reference_model: Optional[str] = None  # If None, uses policy_model as reference
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    load_in_4bit: bool = True
    
    # DPO specific
    beta: float = 0.1  # KL penalty coefficient
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    
    # Training
    output_dir: str = "outputs/dpo"
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    per_device_train_batch_size: int = 1  # 减小以避免 OOM
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32  # 增加以保持 effective batch size
    
    num_train_epochs: int = 1
    max_steps: int = 2000
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10
    
    gradient_checkpointing: bool = False  # 禁用以避免 DDP 兼容性问题
    bf16: bool = True
    
    seed: int = 42
    
    # Data
    max_seq_length: int = 2048
    max_prompt_length: int = 1024


class DPOTrainerWrapper:
    """Wrapper class for DPO training."""
    
    def __init__(self, config: DPOTrainingConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup(self):
        """Setup model and tokenizer."""
        # Load tokenizer
        print(f"Loading tokenizer from: {self.config.policy_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model,
            trust_remote_code=True,
            padding_side="left",  # DPO typically uses left padding
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load policy model with LoRA
        lora_config = None
        if self.config.use_lora:
            lora_config = create_lora_config(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
            )
        
        print("Loading policy model...")
        self.model = load_model_for_training(
            model_name_or_path=self.config.policy_model,
            use_lora=self.config.use_lora,
            load_in_4bit=self.config.load_in_4bit,
            lora_config=lora_config,
            torch_dtype=self.config.torch_dtype,
            attn_implementation=self.config.attn_implementation,
            use_gradient_checkpointing=self.config.gradient_checkpointing,  # 传递配置
        )
        
        # Reference model
        # For DPO, we typically use the same model as reference (frozen)
        # TRL's DPOTrainer handles this automatically when ref_model=None
        self.ref_model = None
        if self.config.reference_model:
            print("Loading reference model...")
            self.ref_model = load_model_for_training(
                model_name_or_path=self.config.reference_model,
                use_lora=False,  # Reference model doesn't need LoRA
                load_in_4bit=self.config.load_in_4bit,
                torch_dtype=self.config.torch_dtype,
            )
    
    def create_training_args(self) -> DPOConfig:
        """Create DPO training arguments."""
        return DPOConfig(
            output_dir=self.config.output_dir,
            
            # DPO specific
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            
            # Optimization
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            optim="adamw_torch",
            
            # Batch size
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Steps
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=3,
            
            # Memory
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # Precision
            bf16=self.config.bf16,
            
            # Data
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_prompt_length,
            
            # Other
            seed=self.config.seed,
            remove_unused_columns=False,
            
            # Logging - 禁用 wandb 避免配置问题
            report_to="none",
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """Run DPO training."""
        if self.model is None or self.tokenizer is None:
            self.setup()
        
        training_args = self.create_training_args()
        
        # Create DPO trainer
        # 新版 TRL 使用 processing_class 代替 tokenizer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        # Train
        print("Starting DPO training...")
        self.trainer.train()
        
        # Save final model
        final_path = os.path.join(self.config.output_dir, "final")
        self.trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Model saved to: {final_path}")
        
        return self.trainer


def create_dpo_trainer(
    config_dict: Dict[str, Any],
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
) -> DPOTrainerWrapper:
    """
    Create DPO trainer from config dictionary.
    
    Args:
        config_dict: Configuration dictionary (from YAML)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Configured DPOTrainerWrapper
    """
    # Flatten nested config
    flat_config = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    config = DPOTrainingConfig(**flat_config)
    wrapper = DPOTrainerWrapper(config)
    
    return wrapper
