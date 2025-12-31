# AI 协作开发项目模板

## 一、项目启动阶段

### 1.1 明确项目需求
向 AI 提供：
- **项目目标**：做什么？（如：微调 Qwen3-4B 模型）
- **硬件约束**：有什么资源？（如：4x H100 GPU, 24h 时间限制）
- **输入输出**：数据是什么？期望输出什么？
- **技术栈偏好**：用什么框架？（如：PyTorch, HuggingFace, DeepSpeed）

### 1.2 让 AI 创建项目骨架
请求 AI：
```
请为我创建完整的项目结构，包括：
- 目录结构
- 配置文件（YAML）
- 核心模块骨架
- README 和依赖文件
```

---

## 二、开发迭代阶段

### 2.1 分阶段开发
将项目拆分为独立阶段，逐个完成：

| 阶段 | 交付物 | 验证标准 |
|------|--------|----------|
| 数据准备 | 数据下载/预处理脚本 | 数据可加载 |
| 模型训练 | 训练脚本 + 配置 | 训练可启动 |
| 模型评估 | 评估脚本 | metrics 可输出 |

### 2.2 高效沟通模式
```
# 报错时的标准格式
报错如下：
[粘贴完整错误日志]

请分析：
1. 哪部分代码导致了崩溃？
2. 错误原因是什么？
3. 如何修复？
```

```
# 请求开发时的格式
请帮我实现 [功能名称]，要求：
- [具体要求1]
- [具体要求2]
- 兼容 [环境/版本]
```

---

## 三、调试与修复阶段

### 3.1 常见问题类型及应对

| 问题类型 | 表现 | 解决思路 |
|----------|------|----------|
| 依赖兼容性 | `TypeError: got unexpected argument` | 检查库版本，适配 API 变化 |
| 显存不足 | `CUDA out of memory` | 减小 batch size，增加 gradient accumulation |
| 分布式训练 | `DDP/DeepSpeed errors` | 检查 device_map，禁用冲突功能 |
| 数据格式 | `Unknown data format` | 打印目录内容，适配多种格式 |

### 3.2 调试流程
1. **粘贴完整错误日志**
2. **让 AI 分析调用栈**
3. **确认修复方案后再执行**
4. **验证修复是否生效**

---

## 四、本项目经验总结

### 4.1 我们解决的关键问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| flash_attn 安装失败 | PyTorch 2.9.1 太新 | 改用 `sdpa` |
| `tokenizer` 参数报错 | 新版 TRL API 变化 | 改为 `processing_class` |
| 4-bit + DDP 冲突 | device_map 不兼容 | 使用 `{"": local_rank}` |
| gradient checkpointing 报错 | 与 DDP + LoRA 冲突 | 禁用或传参控制 |
| OOM | batch size 过大 | 减小 batch，增加 accum |

### 4.2 技术选型建议

```yaml
# 推荐配置（稳定性优先）
model:
  attn_implementation: "sdpa"  # 不要用 flash_attention_2
  load_in_4bit: true           # QLoRA 节省显存

training:
  per_device_batch_size: 1-2   # DPO 时用 1
  gradient_accumulation: 8-32  # 补偿小 batch
  gradient_checkpointing: true # SFT 可用，DPO 需禁用
```

---

## 五、项目启动 Prompt 模板

```markdown
# 项目名称：[名称]

## 项目目标
[一句话描述做什么]

## 硬件环境
- GPU: [型号 x 数量]
- 时间限制: [如有]
- 基座模型路径: [路径]

## 技术要求
- 框架: PyTorch + HuggingFace Transformers
- 训练: DeepSpeed / DDP
- 方法: [SFT/DPO/RLHF]

## 数据集
- 训练: [数据集名称]
- 评估: [benchmark 列表]

## 阶段划分
1. [ ] 环境准备
2. [ ] 数据下载与预处理
3. [ ] 模型训练
4. [ ] 模型评估
5. [ ] 结果分析

请帮我：
1. 创建完整项目结构
2. 编写各阶段的配置和脚本
3. 提供执行命令
```

---

## 六、最佳实践

1. **先跑通再优化**：用小数据/少 epoch 验证流程
2. **保持沟通简洁**：粘贴日志时附带关键行号
3. **分步验证**：每次修改后确认是否解决问题
4. **版本固定**：在 requirements.txt 中固定关键库版本
5. **备份检查点**：训练中间结果及时保存

---

*此模板基于 Qwen3-4B SFT+DPO 微调项目总结，可适配其他 LLM 微调/训练项目。*
