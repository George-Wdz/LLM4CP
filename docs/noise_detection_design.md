# 信道预测与噪声检测模型设计 (Channel Prediction and Noise Detection Model Design)

## 项目概述 (Project Overview)

LLM4CP (Large Language Model for Channel Prediction) 是一个基于GPT-2的信道预测模型，专门用于无线通信系统中的OFDM信道状态信息(CSI)预测。本项目特别关注在噪声和干扰环境下的鲁棒性。

LLM4CP is a GPT-2 based channel prediction model specifically designed for OFDM Channel State Information (CSI) prediction in wireless communication systems. The project particularly focuses on robustness under noise and interference conditions.

## U2D信道预测原理 (U2D Channel Prediction Principle)

### 问题背景 (Problem Background)

在频分双工(FDD)系统中，上行链路(Uplink)和下行链路(Downlink)使用不同的频段，导致信道互易性不成立。传统方法需要下行导频开销来获取下行CSI，但这会降低频谱效率。

In Frequency Division Duplexing (FDD) systems, uplink and downlink use different frequency bands, breaking channel reciprocity. Traditional methods require downlink pilot overhead to obtain downlink CSI, reducing spectral efficiency.

### U2D预测方法 (U2D Prediction Method)

本项目采用两步策略：

1. **时间域预测 (Temporal Prediction)**: 利用历史上行CSI序列 `H_U_his` 预测未来上行CSI `H_U_pre`
2. **频域映射 (Frequency-domain Mapping)**: 学习上行与下行CSI之间的频域映射关系，预测下行CSI `H_D_pre`

The project adopts a two-step strategy:

1. **Temporal Prediction**: Use historical uplink CSI sequence `H_U_his` to predict future uplink CSI `H_U_pre`
2. **Frequency-domain Mapping**: Learn the frequency-domain mapping relationship between uplink and downlink CSI to predict downlink CSI `H_D_pre`

### 模型架构 (Model Architecture)

```
Input: Historical Uplink CSI (16 timesteps × 48 subcarriers)
  ↓
Noise Augmentation (SNR: 5-20 dB random)
  ↓
Jammer Detection Head (Optional)
  ↓
Dual-domain Processing:
  - Delay Domain: IFFT → CNN ResBlocks
  - Frequency Domain: CNN ResBlocks
  ↓
Feature Fusion
  ↓
GPT-2 Encoder (6 layers, frozen except LN/PE)
  ↓
Temporal Projection
  ↓
Output: Predicted CSI (4 future timesteps × 48 subcarriers)
```

## 噪声与干扰处理策略 (Noise and Interference Handling Strategy)

### 1. 训练时噪声增强 (Training-time Noise Augmentation)

**实现位置**: `data.py::noise()` 和 `Dataset_Pro::__init__()`

```python
def noise(H, SNR):
    """Add AWGN to channel matrix"""
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise
```

**策略说明**:
- 每个样本独立添加随机SNR (5-20 dB)的高斯白噪声
- 噪声功率相对于信号功率归一化
- 同时对历史序列和预测目标添加噪声，模拟真实场景

**Strategy**:
- Add AWGN with random SNR (5-20 dB) independently to each sample
- Noise power normalized relative to signal power
- Apply noise to both historical sequence and prediction target to simulate real scenarios

### 2. 干扰感知微调 (Jammer-Aware Fine-Tuning)

**实现位置**: `data.py::apply_jammers()` 和 `models/GPT4CP.py::Model`

支持五种干扰类型：

1. **宽带干扰 (Wideband, 'wb')**: 影响全部子载波
2. **部分带干扰 (Partial-band, 'pb')**: 影响随机选择的连续子载波段
3. **多音干扰 (Multi-tone, 'tone')**: 在多个离散子载波上添加正弦波
4. **脉冲干扰 (Pulsed, 'pulse')**: 时域上的间歇性全频带干扰
5. **跳频干扰 (Frequency-hopping, 'fh')**: 干扰频段随时间跳变

**干扰参数配置** (默认JSR范围: -10 dB to +20 dB):

```json
{
  "jam_prob": 1.0,              // 干扰概率
  "min_types": 1,               // 最少干扰类型数
  "max_types": 2,               // 最多干扰类型数
  "jsr_db_range": [-10.0, 20.0], // JSR范围 (dB)
  "pb_bandwidth_frac_range": [0.1, 0.4],  // 部分带宽度比例
  "tone_count_range": [2, 6],   // 音频数量
  "pulse_duty_cycle_range": [0.05, 0.2],  // 脉冲占空比
  "fh_hop_len_range": [1, 4],   // 跳频时长
  "fh_bandwidth_frac_range": [0.08, 0.2]  // 跳频带宽比例
}
```

### 3. 干扰检测头 (Jammer Detection Head)

**实现位置**: `models/GPT4CP.py::Model.__init__()` 和 `forward()`

```python
# Jam head architecture
self.jam_head = nn.Sequential(
    nn.Linear(jam_input_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, jam_input_dim)
)

# Forward pass with gating
jam_logits = self.jam_head(x_enc)
jam_mask = torch.sigmoid(jam_logits)
x_enc = x_enc * (1.0 - self.jam_gate_strength * jam_mask)
```

**工作原理**:
1. MLP检测头预测每个token的干扰概率 (0-1)
2. 使用sigmoid激活得到软掩码
3. 通过门控机制抑制被干扰的token: `x' = x * (1 - α * mask)`
4. α (jam_gate_strength) 控制抑制强度，范围[0, 1]

**Mechanism**:
1. MLP detection head predicts jammer probability (0-1) for each token
2. Apply sigmoid activation to get soft mask
3. Gate mechanism suppresses jammed tokens: `x' = x * (1 - α * mask)`
4. α (jam_gate_strength) controls suppression strength, range [0, 1]

### 4. 联合损失函数 (Joint Loss Function)

**实现位置**: `train.py::train_loop()`

```python
# Prediction loss (NMSE)
loss_pred = criterion(pred, target)

# Mask detection loss (Binary Cross Entropy)
if use_jammer:
    loss_mask = F.binary_cross_entropy(pred_mask, true_mask)
    total_loss = loss_pred + lambda_mask * loss_mask
else:
    total_loss = loss_pred
```

**损失权重调度**:
- `lambda_mask`: 掩码损失初始权重 (推荐: 1.0-2.0)
- 支持线性调度：从 `lambda_mask` 到 `lambda_mask_final`
- 调度开始epoch: `lambda_mask_switch`
- 调度持续epoch数: `lambda_mask_ramp`

**Loss Weight Scheduling**:
- `lambda_mask`: Initial mask loss weight (recommended: 1.0-2.0)
- Supports linear scheduling: from `lambda_mask` to `lambda_mask_final`
- Schedule start epoch: `lambda_mask_switch`
- Schedule duration: `lambda_mask_ramp` epochs

## 噪声场景下的预测有效性分析 (Prediction Effectiveness Analysis Under Noise)

### 理想下行CSI的作用 (Role of Ideal Downlink CSI)

**问题**: 在噪声和干扰环境下，预测的下行CSI是否仍然有用？

**答案**: 是的，原因如下：

1. **频谱效率提升**: 即使预测CSI有噪声，仍比完全不知道CSI或使用过时CSI要好
2. **导频开销减少**: 可以用粗粒度的导频+预测CSI组合，而非密集导频
3. **鲁棒性设计**: 模型在训练时已见过各种噪声/干扰场景，具有泛化能力

**Question**: Is predicted downlink CSI still useful under noise and interference?

**Answer**: Yes, for the following reasons:

1. **Spectral Efficiency Gain**: Even noisy predicted CSI is better than no CSI or outdated CSI
2. **Pilot Overhead Reduction**: Can combine sparse pilots + predicted CSI instead of dense pilots
3. **Robustness by Design**: Model has seen various noise/interference scenarios during training and generalizes well

### 预测精度 vs. 噪声水平 (Prediction Accuracy vs. Noise Level)

根据论文实验结果 (Figure 7-8):

- **清洁场景** (Clean): NMSE ≈ -20 dB, SE ≈ 5.5 bps/Hz
- **中等噪声** (SNR=10-15dB): NMSE ≈ -15 dB, SE ≈ 5.0 bps/Hz
- **强干扰** (JSR=+10dB): NMSE ≈ -12 dB, SE ≈ 4.5 bps/Hz

性能下降是可接受的，且显著优于基线方法 (PAD, PronyVec)。

According to paper experimental results (Figure 7-8):

- **Clean scenario**: NMSE ≈ -20 dB, SE ≈ 5.5 bps/Hz
- **Moderate noise** (SNR=10-15dB): NMSE ≈ -15 dB, SE ≈ 5.0 bps/Hz
- **Strong interference** (JSR=+10dB): NMSE ≈ -12 dB, SE ≈ 4.5 bps/Hz

Performance degradation is acceptable and significantly better than baseline methods (PAD, PronyVec).

### 实际应用建议 (Practical Application Recommendations)

1. **混合方案** (Hybrid Approach):
   - 使用预测CSI进行预编码初始化
   - 每N个时隙插入稀疏导频进行校准
   - 根据预测置信度动态调整导频密度

2. **干扰检测与回退** (Interference Detection and Fallback):
   - 监控jam_mask输出，检测异常干扰
   - 高干扰场景下增加导频密度或降低调制阶数
   - 使用掩码信息进行选择性子载波调度

3. **自适应训练** (Adaptive Training):
   - 根据实际部署环境的噪声/干扰统计调整训练数据增强参数
   - 使用在线学习或元学习方法持续优化模型

## 使用指南 (Usage Guide)

### 基础训练 (Basic Training)

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --save-path Weights/U2U_LLM4CP.pth \
  --epochs 500 \
  --batch-size 1024 \
  --lr 1e-3
```

### 干扰感知微调 (Jammer-Aware Fine-Tuning)

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --pretrained-path Weights/U2U_LLM4CP.pth \
  --save-path Weights/U2U_LLM4CP_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_moderate.json \
  --lambda-mask 1.0 \
  --jam-gate-strength 1.0 \
  --jam-head-hidden-ratio 0.5 \
  --epochs 100 \
  --batch-size 2048
```

### 自定义干扰配置 (Custom Jammer Configuration)

创建自定义JSON配置文件：

```json
{
  "jam_prob": 0.8,              // 80%的样本添加干扰
  "min_types": 1,
  "max_types": 3,               // 每个样本1-3种干扰类型
  "jsr_db_range": [0.0, 15.0],  // 中等JSR范围
  "pb_bandwidth_frac_range": [0.15, 0.35],
  "tone_count_range": [3, 8],
  "pulse_duty_cycle_range": [0.1, 0.2],
  "fh_hop_len_range": [2, 5],
  "fh_bandwidth_frac_range": [0.1, 0.25]
}
```

使用：`--jammer-cfg configs/my_jammer.json`

### 评估与测试 (Evaluation and Testing)

```bash
# TDD场景测试
python test_tdd_full.py

# FDD场景测试
python test_fdd_full.py
```

测试脚本会生成CSV文件，包含不同速度场景下的NMSE和SE指标。

## 高级优化策略 (Advanced Optimization Strategies)

### 1. 参数高效微调 (Parameter-Efficient Fine-Tuning)

使用LoRA减少可训练参数：

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA rank
    lora_alpha=16,          # Scaling factor
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
    lora_dropout=0.1,
    bias="none"
)

model = Model(...)
model.gpt2 = get_peft_model(model.gpt2, lora_config)
```

### 2. 门控强度预热 (Gate Strength Warmup)

逐步提高干扰抑制强度，避免训练初期不稳定：

```python
# train.py 已实现
current_gate_strength = jam_gate_start + \
    (jam_gate_strength - jam_gate_start) * min(1.0, epoch / jam_gate_warmup)
model.jam_gate_strength = current_gate_strength
```

### 3. 多GPU并行训练 (Multi-GPU Parallel Training)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --multi-gpu \
  --device-ids 0,1,2,3 \
  --batch-size 4096 \
  [other args...]
```

### 4. 清洁验证集评估 (Clean Validation Evaluation)

同时监控干扰场景和清洁场景性能：

```bash
python train.py \
  --use-jammer \
  --eval-clean \
  --log-file logs/training_jam_vs_clean.log \
  [other args...]
```

日志格式：`epoch,nmse_jam,nmse_clean,loss_pred,loss_mask,jam_gate`

## 常见问题 (FAQ)

### Q1: 为什么训练时添加随机噪声？

A: 随机噪声增强提高模型对测试集噪声的鲁棒性。真实环境中信道估计总是有噪声的，训练时模拟这种场景能防止模型过拟合于理想信道。

**Why add random noise during training?**

Random noise augmentation improves model robustness to test-set noise. Real environments always have noisy channel estimation; simulating this during training prevents overfitting to ideal channels.

### Q2: 干扰检测头的hidden_ratio如何选择？

A: 经验值范围 [0.25, 0.75]：
- 更大的ratio (0.75): 更强的检测能力，但参数更多
- 更小的ratio (0.25): 更轻量，适合快速迭代
- 推荐起点: 0.5

**How to choose jam_head hidden_ratio?**

Empirical range [0.25, 0.75]:
- Larger ratio (0.75): Stronger detection but more parameters
- Smaller ratio (0.25): Lighter weight, suitable for rapid iteration
- Recommended starting point: 0.5

### Q3: lambda_mask应该设置多大？

A: 取决于预测质量和检测质量的权衡：
- lambda_mask = 0: 忽略干扰检测，仅优化预测
- lambda_mask = 1.0: 平衡预测和检测 (推荐起点)
- lambda_mask > 2.0: 强调干扰检测
- 建议从1.0开始，观察训练曲线调整

**How to set lambda_mask?**

Depends on the trade-off between prediction and detection quality:
- lambda_mask = 0: Ignore jammer detection, optimize prediction only
- lambda_mask = 1.0: Balance prediction and detection (recommended start)
- lambda_mask > 2.0: Emphasize jammer detection
- Recommend starting from 1.0, observe training curves and adjust

### Q4: 模型在未见过的干扰类型上表现如何？

A: 训练时混合多种干扰类型能提高泛化能力。但对于完全不同的干扰模式（如智能对抗干扰），可能需要重新训练或在线适应。

**How does the model perform on unseen jammer types?**

Training with mixed jammer types improves generalization. However, for completely different interference patterns (e.g., intelligent adversarial jamming), retraining or online adaptation may be needed.

### Q5: 可以只训练干扰检测头，冻结其他参数吗？

A: 可以，通过以下方式：
```python
# 冻结除jam_head外的所有参数
for name, param in model.named_parameters():
    if 'jam_head' not in name:
        param.requires_grad = False
```

这在预训练模型已经很好时可以快速适应新的干扰环境。

**Can I train only the jam_head and freeze other parameters?**

Yes, via:
```python
# Freeze all parameters except jam_head
for name, param in model.named_parameters():
    if 'jam_head' not in name:
        param.requires_grad = False
```

This allows quick adaptation to new interference environments when the pretrained model is already good.

## 参考文献 (References)

1. Liu, B., et al. "LLM4CP: Adapting Large Language Models for Channel Prediction." *Journal of Communications and Information Networks*, 2024.
2. QuaDRiGa Channel Model: https://quadriga-channel-model.de/
3. GPT-2 Architecture: Radford, A., et al. "Language Models are Unsupervised Multitask Learners." *OpenAI*, 2019.

## 更新日志 (Changelog)

- **2025-10-23**: 创建噪声检测与信道预测设计文档
- Initial creation of noise detection and channel prediction design documentation
