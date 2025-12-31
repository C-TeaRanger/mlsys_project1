"""Dataset loading utilities for SFT and DPO training."""

from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer


# ============================================================
# SFT Datasets
# ============================================================

SFT_DATASET_CONFIGS = {
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "split": "train",
        "instruction_col": "instruction",
        "input_col": "input",
        "output_col": "output",
    },
    "sharegpt": {
        "path": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split": "train",
        "conversation_col": "conversations",
    },
    "openorca": {
        "path": "Open-Orca/OpenOrca",
        "split": "train",
        "system_col": "system_prompt",
        "question_col": "question",
        "response_col": "response",
    },
}


def load_sft_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 4096,
    validation_size: float = 0.02,
    seed: int = 42,
    local_data_dir: str = "/data/250010131/codebase/mlsys_project/data/raw",
) -> DatasetDict:
    """
    Load and preprocess SFT dataset.
    
    优先从本地加载数据集，如果本地不存在则从 HuggingFace 下载。
    
    Args:
        dataset_name: Name of the dataset (alpaca, sharegpt, openorca)
        tokenizer: Tokenizer for preprocessing
        max_seq_length: Maximum sequence length
        validation_size: Fraction of data to use for validation
        seed: Random seed for splitting
        local_data_dir: Local directory containing downloaded datasets
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    from datasets import load_from_disk
    import os
    import json
    
    if dataset_name not in SFT_DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(SFT_DATASET_CONFIGS.keys())}")
    
    config = SFT_DATASET_CONFIGS[dataset_name]
    local_path = os.path.join(local_data_dir, dataset_name)
    
    dataset = None
    
    # 尝试从本地加载
    if os.path.exists(local_path):
        print(f"Loading dataset from local: {local_path}")
        
        # 列出目录内容以帮助调试
        contents = os.listdir(local_path)
        print(f"  Directory contents: {contents}")
        
        # 方式1: DatasetDict 格式 (有 dataset_dict.json) - 最常见
        if os.path.exists(os.path.join(local_path, "dataset_dict.json")):
            print("  Detected: DatasetDict format (dataset_dict.json)")
            dataset = load_from_disk(local_path)
            if isinstance(dataset, DatasetDict):
                # 优先使用 train，否则使用第一个 split
                if "train" in dataset:
                    dataset = dataset["train"]
                else:
                    dataset = list(dataset.values())[0]
        
        # 方式2: datasets 格式 (有 dataset_info.json 在根目录)
        elif os.path.exists(os.path.join(local_path, "dataset_info.json")):
            print("  Detected: datasets format (dataset_info.json)")
            dataset = load_from_disk(local_path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset.get("train", list(dataset.values())[0])
        
        # 方式3: 有 train 子目录 (datasets DatasetDict 格式)
        elif os.path.exists(os.path.join(local_path, "train")):
            print("  Detected: datasets format with train/ subdirectory")
            dataset = load_from_disk(local_path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset["train"]
        
        # 方式3: 有 state.json (datasets 格式的另一种标识)
        elif os.path.exists(os.path.join(local_path, "state.json")):
            print("  Detected: datasets format (state.json)")
            dataset = load_from_disk(local_path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset.get("train", list(dataset.values())[0])
        
        # 方式4: JSON 文件格式 (Alpaca)
        elif os.path.exists(os.path.join(local_path, "alpaca_data.json")):
            json_path = os.path.join(local_path, "alpaca_data.json")
            print(f"  Detected: JSON format ({json_path})")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        
        # 方式5: 尝试直接作为 datasets 目录加载
        else:
            print("  Trying to load as datasets directory...")
            try:
                dataset = load_from_disk(local_path)
                if isinstance(dataset, DatasetDict):
                    dataset = dataset.get("train", list(dataset.values())[0])
                print("  Successfully loaded as datasets format")
            except Exception as e:
                print(f"  Failed to load: {e}")
                raise ValueError(f"Unknown data format in {local_path}. Contents: {contents}")
    else:
        # 从 HuggingFace 下载
        print(f"Local data not found, downloading from HuggingFace: {config['path']}")
        dataset = load_dataset(config["path"], split=config["split"])
    
    # Convert to chat format
    if dataset_name == "alpaca":
        dataset = dataset.map(
            lambda x: _format_alpaca(x, config),
            remove_columns=dataset.column_names,
        )
    elif dataset_name == "sharegpt":
        dataset = dataset.map(
            lambda x: _format_sharegpt(x, config),
            remove_columns=dataset.column_names,
        )
    elif dataset_name == "openorca":
        dataset = dataset.map(
            lambda x: _format_openorca(x, config),
            remove_columns=dataset.column_names,
        )
    
    # Apply chat template
    dataset = dataset.map(
        lambda x: _apply_chat_template(x, tokenizer, max_seq_length),
        batched=False,
    )
    
    # Filter out too long sequences
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length)
    print(f"Filtered {original_len - len(dataset)} samples exceeding max_seq_length")
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=validation_size, seed=seed)
    
    return DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"],
    })


def _format_alpaca(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Alpaca format to messages."""
    instruction = example[config["instruction_col"]]
    input_text = example.get(config["input_col"], "")
    output = example[config["output_col"]]
    
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def _format_sharegpt(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ShareGPT format to messages."""
    conversations = example[config["conversation_col"]]
    messages = []
    
    for conv in conversations:
        role = "user" if conv["from"] == "human" else "assistant"
        messages.append({"role": role, "content": conv["value"]})
    
    return {"messages": messages}


def _format_openorca(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenOrca format to messages."""
    messages = []
    
    system = example.get(config["system_col"], "")
    if system:
        messages.append({"role": "system", "content": system})
    
    messages.append({"role": "user", "content": example[config["question_col"]]})
    messages.append({"role": "assistant", "content": example[config["response_col"]]})
    
    return {"messages": messages}


def _apply_chat_template(
    example: Dict[str, Any], 
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
) -> Dict[str, Any]:
    """Apply tokenizer chat template and tokenize."""
    messages = example["messages"]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


# ============================================================
# Preference Datasets (for DPO)
# ============================================================

PREFERENCE_DATASET_CONFIGS = {
    "ultrafeedback": {
        "path": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "prompt_col": "prompt",
        "chosen_col": "chosen",
        "rejected_col": "rejected",
    },
    "helpsteer": {
        "path": "nvidia/HelpSteer",
        "split": "train",
        # Needs special processing
    },
    "anthropic_hh": {
        "path": "Anthropic/hh-rlhf",
        "split": "train",
        "chosen_col": "chosen",
        "rejected_col": "rejected",
    },
}


def load_preference_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    max_prompt_length: int = 1024,
    seed: int = 42,
    local_data_dir: str = "/data/250010131/codebase/mlsys_project/data/raw",
) -> Dataset:
    """
    Load preference dataset for DPO training.
    
    优先从本地加载数据集，如果本地不存在则从 HuggingFace 下载。
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: Tokenizer
        max_seq_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        seed: Random seed
        local_data_dir: Local directory containing downloaded datasets
        
    Returns:
        Dataset with columns: prompt, chosen, rejected
    """
    from datasets import load_from_disk
    import os
    
    if dataset_name not in PREFERENCE_DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = PREFERENCE_DATASET_CONFIGS[dataset_name]
    local_path = os.path.join(local_data_dir, dataset_name)
    
    # 尝试从本地加载
    if os.path.exists(local_path):
        print(f"Loading preference dataset from local: {local_path}")
        dataset = load_from_disk(local_path)
        
        # 获取正确的 split
        if isinstance(dataset, DatasetDict):
            if "train_prefs" in dataset:
                dataset = dataset["train_prefs"]
            elif "train" in dataset:
                dataset = dataset["train"]
            else:
                # 使用第一个可用的 split
                dataset = dataset[list(dataset.keys())[0]]
    else:
        # 从 HuggingFace 下载
        print(f"Local data not found, downloading from HuggingFace: {config['path']}")
        dataset = load_dataset(config["path"], split=config["split"])
    
    # Format dataset to standard columns
    if dataset_name == "ultrafeedback":
        dataset = dataset.map(
            _format_ultrafeedback,
            remove_columns=dataset.column_names,
        )
    elif dataset_name == "anthropic_hh":
        dataset = dataset.map(
            _format_anthropic_hh,
            remove_columns=dataset.column_names,
        )
    
    return dataset


def _format_ultrafeedback(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format UltraFeedback dataset."""
    # UltraFeedback has messages format
    prompt = example["prompt"]
    
    # Get chosen and rejected responses
    chosen = example["chosen"][-1]["content"]  # Last assistant message
    rejected = example["rejected"][-1]["content"]
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def _format_anthropic_hh(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format Anthropic HH dataset."""
    # Parse the conversation format
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # Extract prompt (common prefix) and responses
    # This is a simplified version
    return {
        "prompt": "",  # Needs proper parsing
        "chosen": chosen,
        "rejected": rejected,
    }
