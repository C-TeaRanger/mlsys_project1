#!/bin/bash
# ============================================================
# Qwen3-4B DPO 训练启动脚本
# ============================================================
# 提交作业时的启动命令:
#   source activate /data/250010131/conda_envs/mlsys_project
#   cd /data/250010131/codebase/mlsys_project
#   bash scripts/run_dpo_train.sh
# ============================================================

set -e

echo "=============================================="
echo "Qwen3-4B DPO Training"
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
# 检查 SFT 模型是否存在
# ============================================================
SFT_MODEL_PATH="outputs/sft/final"

if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "错误: SFT 模型不存在: $SFT_MODEL_PATH"
    echo "请先完成 SFT 训练: bash scripts/run_sft_train.sh"
    exit 1
fi

echo "SFT 模型路径: $SFT_MODEL_PATH"
echo ""

# ============================================================
# DPO 训练
# ============================================================
echo "=============================================="
echo "开始 DPO 训练..."
echo "=============================================="

deepspeed --num_gpus=4 scripts/train_dpo.py --config configs/dpo_config.yaml

echo ""
echo "=============================================="
echo "DPO 训练完成!"
echo "结束时间: $(date)"
echo "=============================================="
echo ""
echo "模型保存位置: outputs/dpo/final/"
echo ""
echo "下一步: 运行全量评估"
echo "  python scripts/evaluate.py --model outputs/dpo/final --all"
