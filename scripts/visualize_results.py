#!/usr/bin/env python3
"""
生成评估结果的可视化图表

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --metrics_path report/metrics.json --output_dir report/figures
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# 设置 seaborn 样式
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("paper", font_scale=1.2)

# 设置中文字体（如果需要）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义配色方案
COLORS = {
    'base': '#3498db',   # 蓝色
    'sft': '#2ecc71',    # 绿色
    'dpo': '#e74c3c',    # 红色
    'accent': '#9b59b6'  # 紫色
}


def parse_args():
    parser = argparse.ArgumentParser(description="生成评估结果可视化图表")
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="report/metrics.json",
        help="评估指标 JSON 文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="report/figures",
        help="图表输出目录",
    )
    return parser.parse_args()


def load_metrics(path: str) -> dict:
    """加载评估指标"""
    with open(path, 'r') as f:
        return json.load(f)


def plot_comparison_bar(metrics: dict, output_dir: str):
    """
    生成对比条形图：Base vs SFT vs DPO
    """
    benchmarks = list(metrics['base'].keys())
    x = np.arange(len(benchmarks))
    width = 0.25
    
    base_scores = [metrics['base'][b] * 100 for b in benchmarks]
    sft_scores = [metrics['sft'][b] * 100 for b in benchmarks]
    dpo_scores = [metrics['dpo'][b] * 100 for b in benchmarks]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 使用 seaborn 调色板
    palette = [COLORS['base'], COLORS['sft'], COLORS['dpo']]
    
    bars1 = ax.bar(x - width, base_scores, width, label='Base', color=palette[0], edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x, sft_scores, width, label='SFT', color=palette[1], edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + width, dpo_scores, width, label='DPO', color=palette[2], edgecolor='white', linewidth=0.8)
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Qwen3-4B Fine-tuning Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', ' ').title() for b in benchmarks], fontsize=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='medium')
    
    # 添加背景网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 生成: {output_dir}/comparison_bar.png")


def plot_improvement_chart(metrics: dict, output_dir: str):
    """
    生成改进趋势图：展示 SFT 和 DPO 相对于前一阶段的提升
    """
    benchmarks = list(metrics['base'].keys())
    
    # 计算提升百分比
    sft_improvement = [(metrics['sft'][b] - metrics['base'][b]) * 100 for b in benchmarks]
    dpo_improvement = [(metrics['dpo'][b] - metrics['sft'][b]) * 100 for b in benchmarks]
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_sft = ['#2ecc71' if v >= 0 else '#e74c3c' for v in sft_improvement]
    colors_dpo = ['#27ae60' if v >= 0 else '#c0392b' for v in dpo_improvement]
    
    bars1 = ax.bar(x - width/2, sft_improvement, width, label='SFT vs Base', color=colors_sft, alpha=0.8)
    bars2 = ax.bar(x + width/2, dpo_improvement, width, label='DPO vs SFT', color=colors_dpo, alpha=0.8, hatch='//')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Performance Improvement by Training Stage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{height:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {output_dir}/improvement_chart.png")


def plot_radar_chart(metrics: dict, output_dir: str):
    """
    生成雷达图：多维度能力对比
    """
    benchmarks = list(metrics['base'].keys())
    num_vars = len(benchmarks)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 准备数据
    base_scores = [metrics['base'][b] * 100 for b in benchmarks] + [metrics['base'][benchmarks[0]] * 100]
    sft_scores = [metrics['sft'][b] * 100 for b in benchmarks] + [metrics['sft'][benchmarks[0]] * 100]
    dpo_scores = [metrics['dpo'][b] * 100 for b in benchmarks] + [metrics['dpo'][benchmarks[0]] * 100]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, base_scores, 'o-', linewidth=2, label='Base', color='#3498db')
    ax.fill(angles, base_scores, alpha=0.1, color='#3498db')
    
    ax.plot(angles, sft_scores, 's-', linewidth=2, label='SFT', color='#2ecc71')
    ax.fill(angles, sft_scores, alpha=0.1, color='#2ecc71')
    
    ax.plot(angles, dpo_scores, '^-', linewidth=2, label='DPO', color='#e74c3c')
    ax.fill(angles, dpo_scores, alpha=0.1, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-dimensional Capability Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {output_dir}/radar_chart.png")


def plot_average_performance(metrics: dict, output_dir: str):
    """
    生成平均性能对比图
    """
    models = ['Base', 'SFT', 'DPO']
    
    base_avg = np.mean(list(metrics['base'].values())) * 100
    sft_avg = np.mean(list(metrics['sft'].values())) * 100
    dpo_avg = np.mean(list(metrics['dpo'].values())) * 100
    
    averages = [base_avg, sft_avg, dpo_avg]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(models, averages, color=colors, alpha=0.8, width=0.5)
    
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Average Performance Across All Benchmarks', fontsize=14, fontweight='bold')
    ax.set_ylim(60, 75)
    
    # 添加数值和提升标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if i > 0:
            improvement = averages[i] - averages[i-1]
            ax.annotate(f'(+{improvement:.2f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height + 1.5),
                       ha='center', va='bottom', fontsize=9, color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {output_dir}/average_performance.png")


def plot_training_pipeline(output_dir: str):
    """
    生成训练流程图
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # 绘制流程框
    boxes = [
        (1, 2, 'Qwen3-4B\nBase', '#3498db'),
        (4, 2, 'SFT\n(Alpaca)', '#2ecc71'),
        (7, 2, 'DPO\n(UltraFeedback)', '#e74c3c'),
        (10, 2, 'Final\nModel', '#9b59b6'),
    ]
    
    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 绘制箭头
    arrows = [(2.2, 2), (5.2, 2), (8.2, 2)]
    for x, y in arrows:
        ax.annotate('', xy=(x+1.4, y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # 添加标签
    ax.text(3.5, 1, 'LoRA\n+ Alpaca Data', ha='center', va='center', fontsize=9)
    ax.text(6.5, 1, 'LoRA\n+ Preference Data', ha='center', va='center', fontsize=9)
    ax.text(9.5, 1, 'Merge\nAdapters', ha='center', va='center', fontsize=9)
    
    ax.set_title('Training Pipeline: Qwen3-4B Fine-tuning', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {output_dir}/training_pipeline.png")


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("生成评估结果可视化图表")
    print("=" * 50)
    
    # 加载数据
    metrics = load_metrics(args.metrics_path)
    print(f"加载数据: {args.metrics_path}")
    print()
    
    # 生成所有图表
    plot_comparison_bar(metrics, args.output_dir)
    plot_improvement_chart(metrics, args.output_dir)
    plot_radar_chart(metrics, args.output_dir)
    plot_average_performance(metrics, args.output_dir)
    plot_training_pipeline(args.output_dir)
    
    print()
    print("=" * 50)
    print(f"所有图表已生成到: {args.output_dir}/")
    print("=" * 50)
    print()
    print("生成的图表:")
    print("  1. comparison_bar.png     - 三模型性能对比条形图")
    print("  2. improvement_chart.png  - 训练阶段改进趋势图")
    print("  3. radar_chart.png        - 多维度能力雷达图")
    print("  4. average_performance.png - 平均性能对比图")
    print("  5. training_pipeline.png  - 训练流程示意图")


if __name__ == "__main__":
    main()
