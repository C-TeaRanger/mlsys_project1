#!/usr/bin/env python3
"""
评估结果分析脚本 - 收集并对比 Base/SFT/DPO 模型的评估结果

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results_dir eval_results --output report
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="分析评估结果")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval_results",
        help="评估结果目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report",
        help="输出报告目录",
    )
    return parser.parse_args()


def find_latest_results(model_dir: str) -> dict:
    """找到最新的评估结果文件"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"警告: 目录不存在 {model_dir}")
        return None
    
    # 查找所有 results*.json 文件 (支持嵌套子目录)
    result_files = list(model_path.rglob("results*.json"))
    
    if not result_files:
        print(f"警告: 未找到结果文件 in {model_dir}")
        return None
    
    # 读取最新的结果文件
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"读取结果: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def extract_metrics(results: dict) -> dict:
    """从评估结果中提取指标"""
    if results is None:
        return {}
    
    metrics = {}
    
    # lm-eval 的结果格式
    if "results" in results:
        for task_name, task_results in results["results"].items():
            # 提取主要指标 (通常是 acc 或 acc_norm)
            if "acc,none" in task_results:
                metrics[task_name] = task_results["acc,none"]
            elif "acc_norm,none" in task_results:
                metrics[task_name] = task_results["acc_norm,none"]
            elif "exact_match,none" in task_results:
                metrics[task_name] = task_results["exact_match,none"]
            elif "pass@1,none" in task_results:
                metrics[task_name] = task_results["pass@1,none"]
            else:
                # 取第一个数值指标
                for k, v in task_results.items():
                    if isinstance(v, (int, float)) and not k.startswith("alias"):
                        metrics[task_name] = v
                        break
    
    return metrics


def create_comparison_table(base_metrics: dict, sft_metrics: dict, dpo_metrics: dict) -> list:
    """创建对比表格"""
    all_tasks = set(base_metrics.keys()) | set(sft_metrics.keys()) | set(dpo_metrics.keys())
    
    rows = []
    for task in sorted(all_tasks):
        base_val = base_metrics.get(task, None)
        sft_val = sft_metrics.get(task, None)
        dpo_val = dpo_metrics.get(task, None)
        
        # 计算改进
        sft_improvement = ""
        dpo_improvement = ""
        
        if base_val is not None and sft_val is not None:
            diff = (sft_val - base_val) * 100
            sft_improvement = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        
        if sft_val is not None and dpo_val is not None:
            diff = (dpo_val - sft_val) * 100
            dpo_improvement = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        
        rows.append({
            "task": task,
            "base": f"{base_val*100:.2f}" if base_val else "N/A",
            "sft": f"{sft_val*100:.2f}" if sft_val else "N/A",
            "dpo": f"{dpo_val*100:.2f}" if dpo_val else "N/A",
            "sft_vs_base": sft_improvement,
            "dpo_vs_sft": dpo_improvement,
        })
    
    return rows


def save_csv(rows: list, output_path: str):
    """保存为 CSV 文件"""
    if not rows:
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV 保存到: {output_path}")


def save_markdown_report(rows: list, output_path: str, base_metrics: dict, sft_metrics: dict, dpo_metrics: dict):
    """保存 Markdown 报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Qwen3-4B 微调评估报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 模型对比\n\n")
        f.write("| 模型 | 描述 |\n")
        f.write("|------|------|\n")
        f.write("| Base | Qwen3-4B-Base 原始模型 |\n")
        f.write("| SFT | Supervised Fine-Tuned (Alpaca) |\n")
        f.write("| DPO | Direct Preference Optimization (UltraFeedback) |\n\n")
        
        f.write("## 评估结果\n\n")
        f.write("| Benchmark | Base (%) | SFT (%) | DPO (%) | SFT vs Base | DPO vs SFT |\n")
        f.write("|-----------|----------|---------|---------|-------------|------------|\n")
        
        for row in rows:
            f.write(f"| {row['task']} | {row['base']} | {row['sft']} | {row['dpo']} | {row['sft_vs_base']} | {row['dpo_vs_sft']} |\n")
        
        # 计算平均值
        f.write("\n## 平均性能\n\n")
        
        base_avg = calculate_average(base_metrics)
        sft_avg = calculate_average(sft_metrics)
        dpo_avg = calculate_average(dpo_metrics)
        
        f.write(f"- **Base 平均**: {base_avg*100:.2f}%\n")
        f.write(f"- **SFT 平均**: {sft_avg*100:.2f}%\n")
        f.write(f"- **DPO 平均**: {dpo_avg*100:.2f}%\n\n")
        
        if base_avg > 0 and sft_avg > 0:
            f.write(f"- SFT 相对 Base 提升: **{(sft_avg-base_avg)*100:+.2f}%**\n")
        if sft_avg > 0 and dpo_avg > 0:
            f.write(f"- DPO 相对 SFT 提升: **{(dpo_avg-sft_avg)*100:+.2f}%**\n")
        
        f.write("\n## 结论\n\n")
        f.write("[请根据实际结果填写结论和分析]\n")
    
    print(f"Markdown 报告保存到: {output_path}")


def calculate_average(metrics: dict) -> float:
    """计算平均指标"""
    if not metrics:
        return 0.0
    values = [v for v in metrics.values() if v is not None]
    return sum(values) / len(values) if values else 0.0


def save_raw_json(base_metrics: dict, sft_metrics: dict, dpo_metrics: dict, output_path: str):
    """保存原始 JSON 数据"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "base": base_metrics,
        "sft": sft_metrics,
        "dpo": dpo_metrics,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON 数据保存到: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 50)
    print("评估结果分析")
    print("=" * 50)
    
    # 读取各模型的评估结果
    base_results = find_latest_results(os.path.join(args.results_dir, "base"))
    sft_results = find_latest_results(os.path.join(args.results_dir, "sft"))
    dpo_results = find_latest_results(os.path.join(args.results_dir, "dpo"))
    
    # 提取指标
    base_metrics = extract_metrics(base_results)
    sft_metrics = extract_metrics(sft_results)
    dpo_metrics = extract_metrics(dpo_results)
    
    print(f"\nBase 模型任务数: {len(base_metrics)}")
    print(f"SFT 模型任务数: {len(sft_metrics)}")
    print(f"DPO 模型任务数: {len(dpo_metrics)}")
    
    # 创建对比表格
    rows = create_comparison_table(base_metrics, sft_metrics, dpo_metrics)
    
    # 打印表格
    print("\n" + "=" * 80)
    print("评估结果对比")
    print("=" * 80)
    print(f"{'Benchmark':<20} {'Base':>10} {'SFT':>10} {'DPO':>10} {'SFT↑':>10} {'DPO↑':>10}")
    print("-" * 80)
    for row in rows:
        print(f"{row['task']:<20} {row['base']:>10} {row['sft']:>10} {row['dpo']:>10} {row['sft_vs_base']:>10} {row['dpo_vs_sft']:>10}")
    print("=" * 80)
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    
    save_csv(rows, os.path.join(args.output, "comparison.csv"))
    save_markdown_report(rows, os.path.join(args.output, "evaluation_report.md"), 
                         base_metrics, sft_metrics, dpo_metrics)
    save_raw_json(base_metrics, sft_metrics, dpo_metrics, 
                  os.path.join(args.output, "metrics.json"))
    
    print("\n✓ 分析完成！")
    print(f"报告目录: {args.output}/")


if __name__ == "__main__":
    main()
