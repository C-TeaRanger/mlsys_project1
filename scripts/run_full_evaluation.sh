#!/bin/bash
# ============================================================
# 完整评估脚本：评估 Base / SFT / DPO 三个模型
# ============================================================
# 运行方式（单卡 H100 足够，无需多卡）:
#   source activate /data/250010131/conda_envs/mlsys_project
#   cd /data/250010131/codebase/mlsys_project
#   bash scripts/run_full_evaluation.sh
# ============================================================

set -e

echo "=============================================="
echo "Full Model Evaluation Pipeline"
echo "=============================================="
echo "开始时间: $(date)"
echo ""

# ============================================================
# 环境变量设置
# ============================================================
export CUDA_VISIBLE_DEVICES=0  # 单卡评估即可
export PYTHONPATH="${PYTHONPATH}:/data/250010131/codebase/mlsys_project"

# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/250010131/cache/huggingface

echo "HuggingFace 镜像: $HF_ENDPOINT"
echo "缓存目录: $HF_HOME"

# 输出目录
OUTPUT_DIR="eval_results"
mkdir -p $OUTPUT_DIR

# 模型路径
BASE_MODEL="/data/share/Qwen3-4B-Base"
SFT_ADAPTER="outputs/sft/final"
DPO_ADAPTER="outputs/dpo/final"
SFT_MERGED="outputs/sft/merged"
DPO_MERGED="outputs/dpo/merged"

# 评估任务列表 (5个 benchmark)
# mmlu 缓存格式不兼容 (需要各学科单独配置，但只下载了 'all')
TASKS="hellaswag,winogrande,gsm8k,commonsense_qa,arc_challenge"

# 批次大小 (单卡 H100 80GB 可以用较大 batch)
BATCH_SIZE=16

echo "配置信息:"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - Base 模型: $BASE_MODEL"
echo "  - SFT 适配器: $SFT_ADAPTER"
echo "  - DPO 适配器: $DPO_ADAPTER"
echo "  - 评估任务: $TASKS"
echo "  - 批次大小: $BATCH_SIZE"
echo ""

# ============================================================
# 合并 LoRA 模型 (如果尚未合并)
# ============================================================
if [ ! -f "$SFT_MERGED/config.json" ]; then
    echo "=============================================="
    echo "合并 SFT LoRA 适配器..."
    echo "=============================================="
    python scripts/merge_lora.py \
        --base_model $BASE_MODEL \
        --model $SFT_ADAPTER \
        --output $SFT_MERGED
    echo ""
fi

if [ ! -f "$DPO_MERGED/config.json" ]; then
    echo "=============================================="
    echo "合并 DPO LoRA 适配器..."
    echo "=============================================="
    python scripts/merge_lora.py \
        --base_model $BASE_MODEL \
        --model $DPO_ADAPTER \
        --output $DPO_MERGED
    echo ""
fi

# ============================================================
# 评估 Base 模型 (如果尚未评估)
# ============================================================
if [ -d "$OUTPUT_DIR/base" ] && [ -f "$OUTPUT_DIR/base/results.json" ]; then
    echo "=============================================="
    echo "[1/3] Base 模型已评估，跳过..."
    echo "=============================================="
else
    echo "=============================================="
    echo "[1/3] 评估 Base 模型..."
    echo "=============================================="

    lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL,trust_remote_code=True,dtype=bfloat16 \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        --output_path $OUTPUT_DIR/base \
        --device cuda

    echo ""
    echo "Base 模型评估完成！结果保存在: $OUTPUT_DIR/base"
fi
echo ""

# ============================================================
# 评估 SFT 模型
# ============================================================
echo "=============================================="
echo "[2/3] 评估 SFT 模型..."
echo "=============================================="

lm_eval --model hf \
    --model_args pretrained=$SFT_MERGED,trust_remote_code=True,dtype=bfloat16 \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR/sft \
    --device cuda

echo ""
echo "SFT 模型评估完成！结果保存在: $OUTPUT_DIR/sft"
echo ""

# ============================================================
# 评估 DPO 模型
# ============================================================
echo "=============================================="
echo "[3/3] 评估 DPO 模型..."
echo "=============================================="

lm_eval --model hf \
    --model_args pretrained=$DPO_MERGED,trust_remote_code=True,dtype=bfloat16 \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR/dpo \
    --device cuda

echo ""
echo "DPO 模型评估完成！结果保存在: $OUTPUT_DIR/dpo"
echo ""

# ============================================================
# 汇总结果
# ============================================================
echo "=============================================="
echo "所有评估完成！"
echo "=============================================="
echo "结束时间: $(date)"
echo ""
echo "结果文件位置:"
echo "  - Base: $OUTPUT_DIR/base/"
echo "  - SFT:  $OUTPUT_DIR/sft/"
echo "  - DPO:  $OUTPUT_DIR/dpo/"
echo ""
echo "下一步: 运行结果分析脚本"
echo "  python scripts/analyze_results.py"
