#!/usr/bin/env python3
"""
SFT 训练前检查脚本

运行方法:
    python scripts/pre_training_check.py
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path("/data/250010131/codebase/mlsys_project")


def check_gpu():
    """检查 GPU 可用性"""
    print("=" * 50)
    print("1. GPU 检查")
    print("=" * 50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("  ✗ CUDA 不可用!")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"  ✓ CUDA 可用")
        print(f"  ✓ GPU 数量: {gpu_count}")
        
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {name} ({memory:.1f} GB)")
        
        # 检查 bfloat16 支持
        if torch.cuda.is_bf16_supported():
            print("  ✓ BF16 支持: 是")
        else:
            print("  ⚠ BF16 不支持，将使用 FP16")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GPU 检查失败: {e}")
        return False


def check_libraries():
    """检查关键库版本"""
    print("\n" + "=" * 50)
    print("2. 库版本检查")
    print("=" * 50)
    
    libraries = [
        ("transformers", "4.40.0"),
        ("peft", "0.10.0"),
        ("trl", "0.8.0"),
        ("datasets", "2.18.0"),
        ("accelerate", "0.28.0"),
        ("bitsandbytes", "0.43.0"),
    ]
    
    all_ok = True
    
    for lib_name, min_version in libraries:
        try:
            lib = __import__(lib_name)
            version = lib.__version__
            print(f"  ✓ {lib_name}: {version}")
        except ImportError:
            print(f"  ✗ {lib_name}: 未安装!")
            all_ok = False
        except AttributeError:
            print(f"  ⚠ {lib_name}: 已安装 (无法获取版本)")
    
    # 检查 deepspeed
    try:
        import deepspeed
        print(f"  ✓ deepspeed: {deepspeed.__version__}")
    except ImportError:
        print("  ⚠ deepspeed: 未安装 (可选，用于分布式训练)")
    
    return all_ok


def check_model():
    """检查基座模型"""
    print("\n" + "=" * 50)
    print("3. 基座模型检查")
    print("=" * 50)
    
    model_path = "/data/share/Qwen3-4B-Base"
    
    if not os.path.exists(model_path):
        print(f"  ✗ 模型路径不存在: {model_path}")
        return False
    
    print(f"  ✓ 模型路径存在: {model_path}")
    
    # 检查关键文件
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for f in required_files:
        if os.path.exists(os.path.join(model_path, f)):
            print(f"    ✓ {f}")
        else:
            print(f"    ⚠ {f} 不存在")
    
    # 尝试加载 tokenizer
    try:
        from transformers import AutoTokenizer
        print("  加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  ✓ Tokenizer 加载成功 (vocab_size={len(tokenizer)})")
        
        # 测试 chat template
        test_messages = [{"role": "user", "content": "Hello"}]
        formatted = tokenizer.apply_chat_template(test_messages, tokenize=False)
        print(f"  ✓ Chat Template 可用")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        return False


def check_data():
    """检查数据集"""
    print("\n" + "=" * 50)
    print("4. 数据集检查")
    print("=" * 50)
    
    data_dir = PROJECT_ROOT / "data" / "raw"
    
    datasets_to_check = [
        ("alpaca", "SFT 训练数据"),
        ("ultrafeedback", "DPO 训练数据"),
    ]
    
    all_ok = True
    
    for dataset_name, description in datasets_to_check:
        dataset_path = data_dir / dataset_name
        
        if dataset_path.exists():
            # 统计文件
            files = list(dataset_path.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
            
            print(f"  ✓ {dataset_name} ({description})")
            print(f"    路径: {dataset_path}")
            print(f"    文件数: {file_count}, 大小: {total_size:.1f} MB")
        else:
            print(f"  ✗ {dataset_name} 不存在!")
            print(f"    请运行: python scripts/download_datasets.py")
            all_ok = False
    
    return all_ok


def check_config():
    """检查配置文件"""
    print("\n" + "=" * 50)
    print("5. 配置文件检查")
    print("=" * 50)
    
    import yaml
    
    config_path = PROJECT_ROOT / "configs" / "sft_config.yaml"
    
    if not config_path.exists():
        print(f"  ✗ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        print(f"  ✓ 配置文件加载成功")
        
        # 显示关键配置
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        
        print(f"\n  关键配置:")
        print(f"    base_model: {model_config.get('base_model')}")
        print(f"    lora_r: {model_config.get('lora_r')}")
        print(f"    load_in_4bit: {model_config.get('load_in_4bit')}")
        print(f"    learning_rate: {training_config.get('learning_rate')}")
        print(f"    batch_size: {training_config.get('per_device_train_batch_size')}")
        print(f"    gradient_accumulation: {training_config.get('gradient_accumulation_steps')}")
        print(f"    max_steps: {training_config.get('max_steps')}")
        print(f"    output_dir: {training_config.get('output_dir')}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 配置文件解析失败: {e}")
        return False


def check_output_dir():
    """检查输出目录"""
    print("\n" + "=" * 50)
    print("6. 输出目录检查")
    print("=" * 50)
    
    output_dir = PROJECT_ROOT / "outputs" / "sft"
    
    if output_dir.exists():
        files = list(output_dir.iterdir())
        if files:
            print(f"  ⚠ 输出目录已存在且非空: {output_dir}")
            print(f"    文件数: {len(files)}")
            print(f"    提示: 训练会覆盖已有文件")
        else:
            print(f"  ✓ 输出目录已存在且为空")
    else:
        print(f"  ✓ 输出目录将自动创建: {output_dir}")
    
    return True


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 50)
    print("7. 数据加载测试")
    print("=" * 50)
    
    try:
        # 添加项目路径
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from transformers import AutoTokenizer
        from src.data.dataset import load_sft_dataset
        
        print("  加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/share/Qwen3-4B-Base",
            trust_remote_code=True,
        )
        
        print("  加载 SFT 数据集 (采样 100 条测试)...")
        
        # 尝试加载数据
        from datasets import load_from_disk
        data_path = PROJECT_ROOT / "data" / "raw" / "alpaca"
        dataset = load_from_disk(str(data_path))
        
        if hasattr(dataset, "keys"):
            train_data = dataset.get("train", dataset[list(dataset.keys())[0]])
        else:
            train_data = dataset
        
        sample_count = min(5, len(train_data))
        print(f"  ✓ 数据集加载成功!")
        print(f"    总样本数: {len(train_data)}")
        
        # 显示样本
        print(f"\n  样本预览 (前{sample_count}条):")
        for i in range(sample_count):
            sample = train_data[i]
            if "instruction" in sample:
                text = sample["instruction"][:50] + "..."
            elif "text" in sample:
                text = sample["text"][:50] + "..."
            else:
                text = str(sample)[:50] + "..."
            print(f"    [{i}] {text}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#" * 60)
    print("#  Qwen3-4B SFT 训练前检查")
    print("#" * 60)
    
    results = {}
    
    results["gpu"] = check_gpu()
    results["libraries"] = check_libraries()
    results["model"] = check_model()
    results["data"] = check_data()
    results["config"] = check_config()
    results["output"] = check_output_dir()
    results["data_loading"] = test_data_loading()
    
    # 汇总
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！可以开始 SFT 训练。")
        print("\n启动训练命令:")
        print("  deepspeed --num_gpus=4 scripts/train_sft.py --config configs/sft_config.yaml")
    else:
        print("✗ 部分检查失败，请解决上述问题后再开始训练。")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
