#!/bin/bash
# ============================================================
# Qwen3-4B SFT 训练启动脚本
# ============================================================
# 提交作业时的启动命令:
#   source activate /data/250010131/conda_envs/mlsys_project
#   cd /data/250010131/codebase/mlsys_project
#   bash scripts/run_sft_train.sh
# ============================================================

set -e

echo "=============================================="
echo "Qwen3-4B SFT Training"
echo "=============================================="
echo "开始时间: $(date)"
echo ""

# ============================================================
# 环境变量设置
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:/data/250010131/codebase/mlsys_project"

# Benchmark 缓存路径
export HF_HOME=/data/share/benchmark_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "环境变量已设置"
echo ""

# ============================================================
# 训练前检查
# ============================================================
# echo "运行训练前检查..."
# python scripts/pre_training_check.py

# if [ $? -ne 0 ]; then
#     echo "训练前检查失败，请修复问题后重试"
#     exit 1
# fi
# echo ""

echo "跳过训练前检查"
echo ""

# ============================================================
# SFT 训练
# ============================================================
echo "=============================================="
echo "开始 SFT 训练..."
echo "=============================================="

deepspeed --num_gpus=4 scripts/train_sft.py --config configs/sft_config.yaml

echo ""
echo "=============================================="
echo "SFT 训练完成!"
echo "结束时间: $(date)"
echo "=============================================="
echo ""
echo "模型保存位置: outputs/sft/final/"
echo ""
echo "下一步: 提交 DPO 训练作业"
echo "  bash scripts/run_dpo_train.sh"
