#!/usr/bin/env python3
"""
数据集下载脚本 - 下载 SFT 和 DPO 训练所需的数据集

使用方法:
    # 首先确保安装了 datasets 库
    pip install datasets
    
    # 运行下载脚本
    python scripts/download_datasets.py
    
    # 如果 HuggingFace 访问慢，可以使用镜像
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_datasets.py

数据将保存到:
    - data/raw/alpaca/          - SFT 训练数据 (~52K 样本)
    - data/raw/ultrafeedback/   - DPO 训练数据 (~60K 偏好对)
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path("/data/250010131/codebase/mlsys_project")
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_alpaca():
    """下载 Alpaca 数据集 (用于 SFT)"""
    from datasets import load_dataset
    
    output_path = DATA_DIR / "alpaca"
    
    if output_path.exists():
        print(f"[Alpaca] 数据集已存在: {output_path}")
        return True
    
    print("[Alpaca] 开始下载 SFT 数据集...")
    print("  来源: tatsu-lab/alpaca")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # 保存到本地
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        print(f"[Alpaca] ✓ 下载完成!")
        print(f"  样本数: {len(dataset['train'])}")
        print(f"  保存位置: {output_path}")
        
        # 显示样本
        print("\n  示例数据:")
        sample = dataset['train'][0]
        print(f"    instruction: {sample['instruction'][:80]}...")
        print(f"    output: {sample['output'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"[Alpaca] ✗ 下载失败: {e}")
        return False


def download_ultrafeedback():
    """下载 UltraFeedback 数据集 (用于 DPO)"""
    from datasets import load_dataset
    
    output_path = DATA_DIR / "ultrafeedback"
    
    if output_path.exists():
        print(f"[UltraFeedback] 数据集已存在: {output_path}")
        return True
    
    print("\n[UltraFeedback] 开始下载 DPO 数据集...")
    print("  来源: HuggingFaceH4/ultrafeedback_binarized")
    
    try:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
        
        # 保存到本地
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        print(f"[UltraFeedback] ✓ 下载完成!")
        print(f"  训练集偏好对数: {len(dataset['train_prefs'])}")
        print(f"  保存位置: {output_path}")
        
        # 显示样本
        print("\n  示例数据:")
        sample = dataset['train_prefs'][0]
        print(f"    prompt: {sample['prompt'][:80]}...")
        print(f"    chosen: {str(sample['chosen'])[:80]}...")
        print(f"    rejected: {str(sample['rejected'])[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"[UltraFeedback] ✗ 下载失败: {e}")
        return False


def download_alpaca_json_fallback():
    """备选方案：直接下载 Alpaca JSON 文件"""
    import urllib.request
    import json
    
    output_path = DATA_DIR / "alpaca"
    json_path = output_path / "alpaca_data.json"
    
    if json_path.exists():
        print(f"[Alpaca JSON] 数据已存在: {json_path}")
        return True
    
    print("\n[Alpaca JSON] 尝试直接下载 JSON 文件...")
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(json_path))
        
        # 验证
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"[Alpaca JSON] ✓ 下载完成!")
        print(f"  样本数: {len(data)}")
        print(f"  保存位置: {json_path}")
        
        return True
        
    except Exception as e:
        print(f"[Alpaca JSON] ✗ 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("Qwen3-4B 微调项目 - 数据集下载工具")
    print("=" * 60)
    print(f"\n项目目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print()
    
    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. 下载 Alpaca (SFT)
    print("-" * 40)
    results['alpaca'] = download_alpaca()
    
    # 如果 HuggingFace 方式失败，尝试直接下载 JSON
    if not results['alpaca']:
        print("\n尝试备选下载方案...")
        results['alpaca'] = download_alpaca_json_fallback()
    
    # 2. 下载 UltraFeedback (DPO)
    print("-" * 40)
    results['ultrafeedback'] = download_ultrafeedback()
    
    # 总结
    print("\n" + "=" * 60)
    print("下载结果汇总")
    print("=" * 60)
    print(f"  Alpaca (SFT):       {'✓ 成功' if results['alpaca'] else '✗ 失败'}")
    print(f"  UltraFeedback (DPO): {'✓ 成功' if results['ultrafeedback'] else '✗ 失败'}")
    
    if all(results.values()):
        print("\n✓ 所有数据集下载完成！可以开始训练了。")
        print("\n下一步:")
        print("  1. SFT 阶段: python scripts/train_sft.py --config configs/sft_config.yaml")
        print("  2. DPO 阶段: python scripts/train_dpo.py --config configs/dpo_config.yaml")
    else:
        print("\n⚠ 部分数据集下载失败，请检查网络连接或使用镜像源:")
        print("  HF_ENDPOINT=https://hf-mirror.com python scripts/download_datasets.py")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
