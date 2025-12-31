"""SFT Trainer wrapper for Qwen3-4B fine-tuning."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import os

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from .peft_utils import create_lora_config, load_model_for_training


@dataclass
class SFTTrainingConfig:
    """Configuration for SFT training."""
    
    # Model
    base_model: str = "/data/share/Qwen3-4B-Base"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"  # PyTorch 内置，无需额外安装
    
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
    
    # Training
    output_dir: str = "outputs/sft"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    num_train_epochs: int = 3
    max_steps: int = -1
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    
    gradient_checkpointing: bool = True
    bf16: bool = True
    
    seed: int = 42
    
    # Data
    max_seq_length: int = 4096


class SFTTrainerWrapper:
    """Wrapper class for SFT training."""
    
    def __init__(self, config: SFTTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup(self):
        """Setup model and tokenizer."""
        # Load tokenizer
        print(f"Loading tokenizer from: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with LoRA
        lora_config = None
        if self.config.use_lora:
            lora_config = create_lora_config(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
            )
        
        self.model = load_model_for_training(
            model_name_or_path=self.config.base_model,
            use_lora=self.config.use_lora,
            load_in_4bit=self.config.load_in_4bit,
            lora_config=lora_config,
            torch_dtype=self.config.torch_dtype,
            attn_implementation=self.config.attn_implementation,
        )
    
    def create_training_args(self) -> SFTConfig:
        """Create training arguments."""
        # 检查 TRL 版本，不同版本参数名不同
        import trl
        trl_version = getattr(trl, '__version__', '0.0.0')
        
        config_kwargs = dict(
            output_dir=self.config.output_dir,
            
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
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # Precision
            bf16=self.config.bf16,
            
            # Other
            seed=self.config.seed,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            
            # Logging - 禁用 wandb 避免配置问题
            report_to="none",
        )
        
        return SFTConfig(**config_kwargs)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """Run SFT training."""
        if self.model is None or self.tokenizer is None:
            self.setup()
        
        training_args = self.create_training_args()
        
        # Create trainer
        # 新版 TRL 使用 processing_class 代替 tokenizer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        # Train
        print("Starting SFT training...")
        self.trainer.train()
        
        # Save final model
        final_path = os.path.join(self.config.output_dir, "final")
        self.trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Model saved to: {final_path}")
        
        return self.trainer


def create_sft_trainer(
    config_dict: Dict[str, Any],
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
) -> SFTTrainerWrapper:
    """
    Create SFT trainer from config dictionary.
    
    Args:
        config_dict: Configuration dictionary (from YAML)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Configured SFTTrainerWrapper
    """
    # Flatten nested config
    flat_config = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    config = SFTTrainingConfig(**flat_config)
    wrapper = SFTTrainerWrapper(config)
    
    return wrapper
