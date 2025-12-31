# Qwen3-4B Fine-tuning Project

Fine-tuning Qwen3-4B with **SFT** (Supervised Fine-Tuning) and **DPO** (Direct Preference Optimization).

## 📊 Results

| Model | ARC | CSQA | GSM8K | HellaSwag | WinoGrande | **Average** |
|-------|-----|------|-------|-----------|------------|-------------|
| Base  | 48.1 | 82.7 | 75.4 | 54.5 | 70.3 | 66.2% |
| SFT   | 54.2 | 81.4 | 78.5 | 58.2 | 69.0 | 68.3% |
| DPO   | 54.7 | 79.9 | 79.5 | 61.9 | 67.9 | **68.8%** |

**+2.57% improvement** over base model.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.0+
- 4×H100 80GB (or equivalent)

### Installation

```bash
# Create conda environment
conda create -n mlsys_project python=3.10
conda activate mlsys_project

# Install dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Training

```bash
# Stage 1: SFT Training
bash scripts/run_sft_train.sh

# Stage 2: DPO Training
bash scripts/run_dpo_train.sh
```

### Evaluation

```bash
# Run full evaluation
bash scripts/run_full_evaluation.sh

# Analyze results
python scripts/analyze_results.py
python scripts/visualize_results.py
```

## 📁 Project Structure

```
mlsys_project/
├── configs/                    # Training configurations
│   ├── sft_config.yaml
│   ├── dpo_config.yaml
│   └── ds_config.json
├── scripts/                    # Training & evaluation scripts
│   ├── run_sft_train.sh
│   ├── run_dpo_train.sh
│   ├── run_full_evaluation.sh
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── merge_lora.py
│   ├── analyze_results.py
│   └── visualize_results.py
├── src/                        # Core modules
│   ├── data/                   # Data loading
│   └── training/               # Training logic
├── report/                     # Evaluation reports & figures
│   ├── figures/
│   ├── final_report.md
│   └── evaluation_report.md
└── outputs/                    # Model checkpoints (not in git)
    ├── sft/
    └── dpo/
```

## ⚙️ Training Configuration

### SFT Stage

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-4B-Base |
| Method | QLoRA (4-bit) |
| LoRA Rank | 64 |
| Learning Rate | 2e-4 |
| Batch Size | 128 (effective) |
| Epochs | 3 |
| Dataset | Alpaca (52K) |

### DPO Stage

| Parameter | Value |
|-----------|-------|
| Method | QLoRA (4-bit) |
| LoRA Rank | 64 |
| Learning Rate | 5e-5 |
| Beta | 0.1 |
| Steps | 2000 |
| Dataset | UltraFeedback |

## 📈 Evaluation Benchmarks

| Benchmark | Type | Description |
|-----------|------|-------------|
| ARC-Challenge | Knowledge | Science reasoning |
| CommonsenseQA | Commonsense | Common knowledge QA |
| GSM8K | Math | Grade school math |
| HellaSwag | Commonsense | Sentence completion |
| WinoGrande | Commonsense | Pronoun resolution |

## 🛠️ Key Features

- **QLoRA**: 4-bit quantization + LoRA for memory efficiency
- **DeepSpeed ZeRO-2**: Distributed training optimization
- **SDPA**: Scaled dot-product attention (PyTorch native)
- **Two-stage Training**: SFT → DPO pipeline

## 📝 Reports

- [Final Report](report/final_report.md) - Complete experiment analysis
- [Evaluation Report](report/evaluation_report.md) - Benchmark results

## 🔧 Troubleshooting

### OOM Error
Reduce batch size in config:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
```

### Lock File Permission
Set user-writable HF cache:
```bash
export HF_HOME=/path/to/your/cache
```

## 📜 License

MIT License

## 🙏 Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) - Base model
- [HuggingFace TRL](https://github.com/huggingface/trl) - Training framework
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Evaluation
