# LLM4CP实战指南 - 噪声与干扰场景处理

## 目录

1. [快速开始](#快速开始)
2. [噪声场景实验](#噪声场景实验)
3. [干扰场景实验](#干扰场景实验)
4. [性能调优](#性能调优)
5. [故障排查](#故障排查)
6. [最佳实践](#最佳实践)

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
conda create -n llm4cp python=3.8
conda activate llm4cp

# 安装依赖
cd LLM4CP
pip install -r requirements.txt

# 下载并缓存GPT-2权重（可选但推荐）
mkdir -p hf_models/gpt2
cd hf_models/gpt2
wget https://hf-mirror.com/gpt2/resolve/main/config.json
wget https://hf-mirror.com/gpt2/resolve/main/merges.txt
wget https://hf-mirror.com/gpt2/resolve/main/vocab.json
wget https://hf-mirror.com/gpt2/resolve/main/pytorch_model.bin
cd ../..

# 设置环境变量
export GPT2_LOCAL_PATH=$(pwd)/hf_models/gpt2
export TRANSFORMERS_OFFLINE=1
```

### 数据准备

确保以下目录结构：

```
LLM4CP/
├── Dataset/
│   ├── train_data/
│   │   ├── H_U_his_train.mat  # 训练集历史上行CSI
│   │   ├── H_U_pre_train.mat  # 训练集未来上行CSI
│   │   └── H_D_pre_train.mat  # 训练集未来下行CSI（FDD场景）
│   └── test_data/
│       ├── H_U_his_test.mat   # 测试集历史上行CSI
│       ├── H_U_pre_test.mat   # 测试集未来上行CSI
│       └── H_D_pre_test.mat   # 测试集未来下行CSI
└── Weights/
    └── U2U_LLM4CP.pth         # 预训练权重（可选）
```

## 噪声场景实验

### 实验1: 基础噪声鲁棒性训练

**目标**: 训练一个对不同SNR鲁棒的基础模型

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --save-path Weights/U2U_LLM4CP_noise_robust.pth \
  --epochs 500 \
  --batch-size 1024 \
  --lr 1e-3 \
  --lr-step-size 150 \
  --lr-gamma 0.1 \
  --device cuda:0
```

**关键点**:
- `data.py`中的`noise()`函数自动为每个样本添加SNR∈[5,20]dB的噪声
- 训练集和验证集都会经过噪声增强
- 模型学习提取噪声中的信号特征

**预期结果**:
- 训练NMSE收敛到 -18 ~ -20 dB
- 验证NMSE收敛到 -16 ~ -18 dB
- 约需要300-400 epochs达到最佳性能

### 实验2: 评估不同SNR下的性能

修改`test_tdd_full.py`添加自定义噪声水平：

```python
# 在测试脚本中添加噪声扫描
SNR_levels = [5, 10, 15, 20, 25, 30]  # dB
results = {snr: {'nmse': [], 'se': []} for snr in SNR_levels}

for snr in SNR_levels:
    # 添加噪声到测试数据
    test_data_prev_noisy = noise(test_data_prev, snr)
    test_data_pred_noisy = noise(test_data_pred, snr)
    
    # 运行预测和评估
    # ... (使用noisy数据)
    
    results[snr]['nmse'].append(nmse_value)
    results[snr]['se'].append(se_value)

# 绘制性能曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(SNR_levels, [np.mean(results[snr]['nmse']) for snr in SNR_levels])
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('NMSE vs SNR')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(SNR_levels, [np.mean(results[snr]['se']) for snr in SNR_levels])
plt.xlabel('SNR (dB)')
plt.ylabel('Spectral Efficiency (bps/Hz)')
plt.title('SE vs SNR')
plt.grid(True)

plt.tight_layout()
plt.savefig('snr_performance_curve.png', dpi=300)
```

## 干扰场景实验

### 实验3: 单一干扰类型微调

**场景**: 专门对抗部分带干扰

创建配置文件 `configs/jammer_pb_only.json`:

```json
{
  "types": ["pb"],
  "min_types": 1,
  "max_types": 1,
  "jam_prob": 1.0,
  "jsr_db_range": [-10.0, 20.0],
  "pb_bandwidth_frac_range": [0.2, 0.5],
  "num_subcarriers": 48
}
```

训练命令：

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --pretrained-path Weights/U2U_LLM4CP.pth \
  --save-path Weights/U2U_LLM4CP_pb_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_pb_only.json \
  --lambda-mask 1.5 \
  --jam-gate-strength 1.0 \
  --jam-gate-start 0.0 \
  --jam-gate-warmup 50 \
  --jam-head-hidden-ratio 0.5 \
  --jam-head-lr-mult 2.0 \
  --epochs 100 \
  --batch-size 2048 \
  --lr 5e-4
```

**训练策略解析**:
- `--jam-gate-start 0.0`: 初始门控强度为0（不抑制）
- `--jam-gate-warmup 50`: 在前50个epoch逐步提高到1.0
- `--jam-head-lr-mult 2.0`: 干扰检测头学习率是基础学习率的2倍
- `--lr 5e-4`: 微调时使用较小学习率

### 实验4: 混合干扰微调

**场景**: 应对真实环境中的多种干扰

使用预设的 `configs/jammer_moderate.json`:

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --pretrained-path Weights/U2U_LLM4CP.pth \
  --save-path Weights/U2U_LLM4CP_mixed_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_moderate.json \
  --lambda-mask 1.0 \
  --lambda-mask-final 2.0 \
  --lambda-mask-switch 30 \
  --lambda-mask-ramp 20 \
  --jam-gate-strength 0.8 \
  --jam-gate-warmup 30 \
  --jam-head-hidden-ratio 0.75 \
  --epochs 150 \
  --batch-size 1024 \
  --eval-clean \
  --log-file logs/mixed_jam_training.log
```

**高级功能**:
- `--lambda-mask-final 2.0`: 从1.0逐步提高到2.0
- `--lambda-mask-switch 30`: 从第30个epoch开始调整
- `--lambda-mask-ramp 20`: 用20个epoch完成过渡
- `--eval-clean`: 同时评估清洁场景性能
- `--log-file`: 记录详细训练日志

**日志分析**:

```bash
# 查看训练日志
tail -f logs/mixed_jam_training.log

# 日志格式：epoch,nmse_jam,nmse_clean,loss_pred,loss_mask,jam_gate
# 使用Python分析日志
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/mixed_jam_training.log', 
                 names=['epoch', 'nmse_jam', 'nmse_clean', 'loss_pred', 'loss_mask', 'jam_gate'])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# NMSE对比
axes[0, 0].plot(df['epoch'], df['nmse_jam'], label='Jammed')
axes[0, 0].plot(df['epoch'], df['nmse_clean'], label='Clean')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('NMSE (dB)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 损失曲线
axes[0, 1].plot(df['epoch'], df['loss_pred'], label='Prediction Loss')
axes[0, 1].plot(df['epoch'], df['loss_mask'], label='Mask Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 门控强度
axes[1, 0].plot(df['epoch'], df['jam_gate'])
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Gate Strength')
axes[1, 0].grid(True)

# 性能差距
axes[1, 1].plot(df['epoch'], df['nmse_jam'] - df['nmse_clean'])
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Performance Gap (dB)')
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300)
```

### 实验5: 极端干扰场景

**场景**: 强干扰环境 (JSR up to +30 dB)

创建 `configs/jammer_extreme.json`:

```json
{
  "types": ["wb", "pb", "tone", "pulse", "fh"],
  "min_types": 2,
  "max_types": 4,
  "jam_prob": 1.0,
  "jsr_db_range": [0.0, 30.0],
  "pb_bandwidth_frac_range": [0.3, 0.7],
  "tone_count_range": [5, 15],
  "pulse_duty_cycle_range": [0.1, 0.3],
  "fh_hop_len_range": [1, 3],
  "fh_bandwidth_frac_range": [0.15, 0.35],
  "num_subcarriers": 48
}
```

训练：

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --pretrained-path Weights/U2U_LLM4CP_mixed_jam.pth \
  --save-path Weights/U2U_LLM4CP_extreme_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_extreme.json \
  --lambda-mask 2.5 \
  --jam-gate-strength 1.0 \
  --jam-head-hidden-ratio 0.75 \
  --epochs 200 \
  --batch-size 512 \
  --lr 1e-4 \
  --multi-gpu \
  --device-ids 0,1,2,3
```

## 性能调优

### 策略1: 学习率调优

不同场景的推荐学习率：

| 场景 | 初始学习率 | 学习率衰减 | 备注 |
|------|-----------|-----------|------|
| 从头训练 | 1e-3 | StepLR(150, 0.1) | 标准配置 |
| 微调（轻度干扰） | 5e-4 | StepLR(50, 0.5) | 保持预训练知识 |
| 微调（强干扰） | 1e-4 | 无 | 小心调整 |
| 仅jam_head | 1e-3 | 无 | 冻结其他参数 |

### 策略2: Batch Size vs. GPU显存

| GPU | 显存 | 推荐Batch Size | 备注 |
|-----|------|---------------|------|
| RTX 3090 | 24GB | 2048-4096 | 单卡全模型训练 |
| RTX 4090 | 24GB | 4096-8192 | 更快训练 |
| V100 | 32GB | 8192 | 大batch稳定 |
| A100 | 40GB/80GB | 16384+ | 最大batch |

显存不足时：
- 减小batch size
- 启用梯度累积 (需要修改train.py)
- 使用LoRA参数高效微调

### 策略3: 超参数网格搜索

创建搜索脚本 `scripts/hyperparam_search.sh`:

```bash
#!/bin/bash

for lambda in 0.5 1.0 1.5 2.0; do
  for gate in 0.5 0.7 0.9 1.0; do
    for hidden in 0.25 0.5 0.75; do
      echo "Training with lambda=$lambda, gate=$gate, hidden=$hidden"
      
      python train.py \
        --train-r-path Dataset/train_data/H_U_his_train.mat \
        --train-t-path Dataset/train_data/H_U_pre_train.mat \
        --pretrained-path Weights/U2U_LLM4CP.pth \
        --save-path Weights/search_l${lambda}_g${gate}_h${hidden}.pth \
        --use-jammer \
        --lambda-mask $lambda \
        --jam-gate-strength $gate \
        --jam-head-hidden-ratio $hidden \
        --epochs 50 \
        --batch-size 2048 \
        --log-file logs/search_l${lambda}_g${gate}_h${hidden}.log
        
    done
  done
done

# 分析结果
python scripts/analyze_search_results.py logs/search_*.log
```

## 故障排查

### 问题1: 训练损失不收敛

**症状**: Loss在高值震荡，不下降

**可能原因**:
1. 学习率过大
2. Batch size过小导致梯度不稳定
3. 干扰参数过于激进

**解决方案**:
```bash
# 降低学习率
--lr 1e-4  # 原来是1e-3

# 增大batch size
--batch-size 4096  # 原来是1024

# 使用更温和的干扰配置
--jammer-cfg configs/jammer_moderate.json  # 而非jammer_dense.json

# 增加gate warmup
--jam-gate-warmup 100  # 原来是50
```

### 问题2: 验证性能比训练性能差很多

**症状**: 训练NMSE -18dB，验证NMSE -10dB

**可能原因**:
1. 过拟合
2. 训练/验证数据分布不匹配
3. Batch normalization问题

**解决方案**:
```bash
# 增加正则化
--weight-decay 1e-3  # 原来是1e-4

# 使用更多数据增强
# 修改data.py中的噪声范围：
# random.rand() * 15 + 5.0  → random.rand() * 20 + 5.0

# 早停策略（修改train.py）
# 添加early stopping逻辑
```

### 问题3: 干扰检测效果差

**症状**: Mask BCE loss很高，预测mask与真实mask差异大

**可能原因**:
1. lambda_mask设置过小
2. jam_head网络容量不足
3. 门控强度上升过快

**解决方案**:
```bash
# 提高mask loss权重
--lambda-mask 2.0  # 原来是1.0

# 增大jam_head容量
--jam-head-hidden-ratio 0.75  # 原来是0.5

# 延长warmup
--jam-gate-warmup 150  # 原来是50

# 提高jam_head学习率
--jam-head-lr-mult 3.0  # 原来是2.0
```

### 问题4: GPU内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:

```bash
# 方案1: 减小batch size
--batch-size 512  # 原来是2048

# 方案2: 使用梯度检查点（需要修改代码）
# 在models/GPT4CP.py中：
# self.gpt2.gradient_checkpointing_enable()

# 方案3: 使用LoRA
# 见下一节

# 方案4: 使用多GPU
--multi-gpu --device-ids 0,1
```

### 问题5: 多GPU训练速度慢

**症状**: 4块GPU比1块GPU慢

**可能原因**:
1. Batch size太小，通信开销大
2. DataLoader workers不足
3. GPU间负载不均

**解决方案**:
```bash
# 增大batch size
--batch-size 8192  # 每块GPU分到2048

# 增加数据加载workers
--num-workers 32  # 原来是16

# 检查GPU使用率
watch -n 1 nvidia-smi
```

## 最佳实践

### 1. 渐进式训练策略

**阶段1: 基础训练（无干扰）**
```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --save-path Weights/stage1_base.pth \
  --epochs 300 \
  --batch-size 2048
```

**阶段2: 轻度干扰微调**
```bash
python train.py \
  --pretrained-path Weights/stage1_base.pth \
  --save-path Weights/stage2_light_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_light.json \
  --lambda-mask 0.5 \
  --epochs 100
```

**阶段3: 强干扰适应**
```bash
python train.py \
  --pretrained-path Weights/stage2_light_jam.pth \
  --save-path Weights/stage3_heavy_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_dense.json \
  --lambda-mask 1.5 \
  --epochs 100
```

### 2. 使用LoRA高效微调

修改训练脚本启用LoRA：

```python
# train.py 中添加
from peft import get_peft_model, LoraConfig, TaskType

# 在模型创建后
if args.use_lora:
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    model.gpt2 = get_peft_model(model.gpt2, lora_config)
    model.gpt2.print_trainable_parameters()
```

使用：
```bash
python train.py \
  --use-lora \
  --use-jammer \
  --batch-size 8192  # LoRA显存占用更小
  [other args...]
```

### 3. 监控与可视化

使用TensorBoard（需修改train.py添加）：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_name')

# 在训练循环中
writer.add_scalar('Loss/train_pred', loss_pred, epoch)
writer.add_scalar('Loss/train_mask', loss_mask, epoch)
writer.add_scalar('NMSE/train', nmse_train, epoch)
writer.add_scalar('NMSE/val_jammed', nmse_val_jam, epoch)
writer.add_scalar('NMSE/val_clean', nmse_val_clean, epoch)
writer.add_scalar('Params/jam_gate_strength', current_gate, epoch)

# 每10个epoch记录一次掩码可视化
if epoch % 10 == 0:
    writer.add_images('Masks/predicted', pred_mask[:4].unsqueeze(1), epoch)
    writer.add_images('Masks/ground_truth', true_mask[:4].unsqueeze(1), epoch)

writer.close()
```

启动TensorBoard：
```bash
tensorboard --logdir runs/ --port 6006
# 浏览器访问 http://localhost:6006
```

### 4. 模型性能评估清单

完整评估流程：

```bash
# 1. 清洁场景
python test_tdd_full.py  # 使用基础模型

# 2. 不同噪声水平
for snr in 5 10 15 20 25; do
  python test_with_noise.py --snr $snr --model Weights/U2U_LLM4CP.pth
done

# 3. 不同干扰类型
for jammer in wb pb tone pulse fh; do
  python test_with_jammer.py \
    --jammer-type $jammer \
    --jsr 10 \
    --model Weights/U2U_LLM4CP_jam.pth
done

# 4. 不同JSR水平
for jsr in -10 -5 0 5 10 15 20; do
  python test_with_jammer.py \
    --jammer-cfg configs/jammer_moderate.json \
    --jsr $jsr \
    --model Weights/U2U_LLM4CP_jam.pth
done

# 5. 生成性能报告
python scripts/generate_report.py \
  --results-dir results/ \
  --output report.pdf
```

### 5. 部署前检查

在实际部署前完成以下检查：

- [ ] 模型在多个速度场景下测试 (3km/h, 30km/h, 120km/h)
- [ ] 验证计算延迟满足实时要求 (<5ms per batch)
- [ ] 测试不同天线配置的泛化性能
- [ ] 评估长时间运行的稳定性
- [ ] 准备模型回退方案（如检测到异常高NMSE时切换到传统方法）
- [ ] 导出ONNX或TorchScript用于生产环境

```python
# 导出模型
import torch

model = torch.load('Weights/U2U_LLM4CP_jam.pth')
model.eval()

# 示例输入
dummy_input = torch.randn(1, 16, 96).to(device)
dummy_mark = torch.zeros(1, 16, 4).to(device)

# 导出为TorchScript
traced_model = torch.jit.trace(model, (dummy_input, dummy_mark, None, None))
traced_model.save('deploy/llm4cp_model.pt')

# 或导出为ONNX
torch.onnx.export(
    model,
    (dummy_input, dummy_mark, None, None),
    'deploy/llm4cp_model.onnx',
    input_names=['input', 'mark'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## 总结

本指南涵盖了LLM4CP在噪声和干扰场景下的实战应用。关键要点：

1. **渐进式训练**: 从清洁场景到噪声场景再到干扰场景
2. **仔细调参**: lambda_mask, gate_strength, hidden_ratio需要根据具体场景调整
3. **充分监控**: 使用日志和可视化工具追踪训练过程
4. **全面测试**: 在多种场景下评估模型鲁棒性
5. **持续优化**: 根据实际部署反馈迭代改进

有问题欢迎提Issue或邮件联系作者！
