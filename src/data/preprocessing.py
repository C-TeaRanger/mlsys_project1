"""Data preprocessing utilities."""

from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer


def preprocess_sft_data(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 4096,
    add_special_tokens: bool = True,
) -> Dict[str, List]:
    """
    Preprocess SFT data with chat template.
    
    This function is designed to be used with dataset.map() in batched mode.
    
    Args:
        examples: Batch of examples with 'messages' column
        tokenizer: Tokenizer
        max_seq_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens
        
    Returns:
        Tokenized examples
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for messages in examples["messages"]:
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
            add_special_tokens=add_special_tokens,
            return_tensors=None,
        )
        
        all_input_ids.append(tokenized["input_ids"])
        all_attention_mask.append(tokenized["attention_mask"])
        all_labels.append(tokenized["input_ids"].copy())
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply chat template to messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Tokenizer with chat template
        add_generation_prompt: Whether to add generation prompt
        
    Returns:
        Formatted text string
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def create_prompt_completion_pairs(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 4096,
) -> Dict[str, List]:
    """
    Create prompt-completion pairs for training.
    
    This masks the prompt tokens in labels so loss is only computed on completions.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer
        max_seq_length: Maximum sequence length
        
    Returns:
        Tokenized examples with masked labels
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    IGNORE_INDEX = -100  # CrossEntropyLoss ignore index
    
    for messages in examples["messages"]:
        # Separate user and assistant messages
        prompt_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                break
            prompt_messages.append(msg)
        
        # Tokenize prompt only (to get prompt length)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokenized = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_tokenized["input_ids"])
        
        # Tokenize full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]
        
        # Create labels: mask prompt tokens
        labels = input_ids.copy()
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
        
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def get_data_collator(tokenizer: PreTrainedTokenizer, padding: str = "longest"):
    """
    Get data collator for training.
    
    Args:
        tokenizer: Tokenizer
        padding: Padding strategy
        
    Returns:
        DataCollator
    """
    from transformers import DataCollatorForSeq2Seq
    
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=padding,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
