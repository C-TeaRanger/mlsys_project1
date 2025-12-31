# Qwen3-4B 微调评估报告

生成时间: 2025-12-31 14:02:04

## 模型对比

| 模型 | 描述 |
|------|------|
| Base | Qwen3-4B-Base 原始模型 |
| SFT | Supervised Fine-Tuned (Alpaca) |
| DPO | Direct Preference Optimization (UltraFeedback) |

## 评估结果

| Benchmark | Base (%) | SFT (%) | DPO (%) | SFT vs Base | DPO vs SFT |
|-----------|----------|---------|---------|-------------|------------|
| arc_challenge | 48.12 | 54.18 | 54.69 | +6.1% | +0.5% |
| commonsense_qa | 82.72 | 81.41 | 79.93 | -1.3% | -1.5% |
| gsm8k | 75.44 | 78.54 | 79.53 | +3.1% | +1.0% |
| hellaswag | 54.52 | 58.16 | 61.92 | +3.6% | +3.8% |
| winogrande | 70.32 | 68.98 | 67.88 | -1.3% | -1.1% |

## 平均性能

- **Base 平均**: 66.22%
- **SFT 平均**: 68.25%
- **DPO 平均**: 68.79%

- SFT 相对 Base 提升: **+2.03%**
- DPO 相对 SFT 提升: **+0.54%**

## 结论

[请根据实际结果填写结论和分析]
