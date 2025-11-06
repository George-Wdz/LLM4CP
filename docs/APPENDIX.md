# 附录：LLM4CP 训练与模型说明（含 Jammer Head 公式）

本文档面向论文撰写与技术归档，系统性整理本项目的数据/符号、模型结构、Jammer Head（检测与掩蔽）、训练目标与调度、指标与超参映射等内容。

---

## A.1 符号与张量尺寸

- B：batch 大小
- L：历史长度（`prev_len`，默认 16）
- T：预测长度（`pred_len`，默认 4）
- K：子载波数（典型 48/64，数据与配置需一致）
- Nt = UQh·UQv，Nr = BQh·BQv：发/收天线数（默认 1×1）
- enc_in = K·Nt·Nr；c_out = enc_in；D = 2·enc_in（实虚拼接后维度）
- 输入 X ∈ R^{B×L×D}，目标 Y ∈ R^{B×T×D}，预测 Ŷ ∈ R^{B×T×D}

注：实虚拼接约定为最后一维按 [Re, Im] 连接；在 `data.py` 的 `LoadBatch_ofdm*` 中完成复数→实数展开。

---

## A.2 数据生成与预处理（`data.py`）

### A.2.1 AWGN（默认开启）

- 随机选取 SNR ∈ [5, 20] dB（实现：`random.rand()*15+5`）。
- 加噪公式（对历史与目标都加）：
  $$ H_{awgn} = H + \sqrt{\tfrac{\sigma}{2}}(N_r + j N_i),\quad \sigma = 10^{-\mathrm{SNR}/10} $$
  其中噪声按样本能量做匹配缩放：$\|H\|$ 标定见 `noise()`。

### A.2.2 合成干扰与监督掩码 jam_mask（仅输入侧）

- 配置（可通过 JSON 覆盖，参考 `configs/jammer_*.json`）：
  - `types`∈{wb, pb, tone, pulse, fh}，`jam_prob`∈[0,1]，`min_types`/`max_types`
  - `jsr_db_range = [J_{min}, J_{max}]`
  - `pb_bandwidth_frac_range`、`tone_count_range`、`pulse_duty_cycle_range`、`fh_hop_len_range`、`fh_bandwidth_frac_range`
  - `num_subcarriers = K`
- 统一强度标定（对每类干扰）：
  $$ \sigma_{jam} = \sqrt{\frac{\mathrm{mean}(|H|^2) \cdot 10^{\mathrm{JSR}/10}}{2}} $$
- 类型简述：
  - 宽带 wb：全带随机噪声（掩码全 1）。
  - 部分带宽 pb：选择子载波子集 S 覆盖并置掩码 1。
  - 多音 tone：在若干子载波上叠加正弦项并置掩码 1。
  - 脉冲 pulse：沿时间选占空窗口加干扰并置掩码 1。
  - 跳频 fh：以 hop 长度在时间上跳变的带宽子集上加干扰并置掩码 1。

### A.2.3 规范化与打包

- 归一化：$x = (x-\mathrm{mean}(x))/\mathrm{std}(x)$（批内标量）。
- OFDM 展开：将复数通道按 `LoadBatch_ofdm*` 转为实数张量（形状见 A.1 约定）。

---

## A.3 模型结构（`models/GPT4CP.py`）

整体前向流程：

1. 输入标准化 X；
2. （可选）Jammer Head 产生掩码并做软门控：$X\to \tilde{X}$；
3. 双域残差预处理：时延域 RB_f + 频域 RB_e → 相加；
4. 嵌入与对齐至 GPT-2 维度；
5. GPT-2 前若干层（截断使用）；
6. 线性还原通道维并通过时间头预测未来 T 步；
7. 反归一化得到 Ŷ。

### A.3.1 Jammer Head（检测与掩蔽，重点）

- 结构：两层 MLP（Linear→GELU→Linear），输入/输出维均为 D，隐藏维为 r·D（r=`jam_head_hidden_ratio`）。
- 概率掩码（逐时间×逐特征）：
  $$ M = \sigma\big(\mathrm{MLP}(X)\big),\quad M \in [0,1]^{B\times L\times D} $$
- 乘性软门控（抑制强度 g ∈ [0,1]）：
  $$ \tilde{X} = X \odot (1 - g\cdot M) $$
  - g 越大抑制越强；g=0 表示不抑制（只检测不改输入）。
  - 放置位置：标准化之后、进入双域残差块之前。
- 训练时同时最小化主任务与掩码任务（见 A.4），推理时可按场景将 g 设为 0（干净场景）或中等（存在干扰）。

### A.3.2 双域残差、嵌入与 GPT-2 主干

- 双域：`RB_f`（时延域，含 IFFT 复原与残差卷积）和 `RB_e`（频域残差卷积），两路相加。
- 嵌入：`DataEmbedding` 将特征对齐到 `d_model` 并加位置编码；再映射到 GPT 输入维（`gpt_dim`）。
- GPT-2：截取前 `gpt_layers` 层 `last_hidden_state`；
- 输出层：`Linear(d_ff→2·c_out)` + 时间头 `Linear(L→T)`，再反归一化。

---

## A.4 训练目标与调度（`train.py`）

### A.4.1 损失函数

- 主任务（信道预测）NMSE（逐样本，再取均值）：
  $$ \mathrm{NMSE}(\hat{Y},Y) = \frac{\|\hat{Y}-Y\|_2^2}{\|Y\|_2^2 + \varepsilon} $$
- 掩码（仅 `--use-jammer`）：BCE
  $$ L_{mask} = -\,\mathrm{mean}\,[\, m\log p + (1-m)\log(1-p)\,],\quad p=M,\ m=\mathrm{jam\_mask} $$
- 总损失：
  $$ L_{total} = L_{pred} + \lambda_{mask}\, L_{mask} $$

### A.4.2 调度（Scheduling）

- 门控强度（线性 warmup，W 为预热轮次）：
  $$ g(e) = g_{start} + (g_{target}-g_{start})\cdot \mathrm{clip}(e/W,0,1) $$
- \(\lambda_{mask}\)（分段/斜坡；S 为切换轮次，R 为斜坡长度）：
  - 阶跃：$ e<S: \lambda=\lambda_0;\ e\ge S: \lambda=\lambda_F$
  - 斜坡：$ \lambda(e)=\lambda_0+(\lambda_F-\lambda_0)\cdot \mathrm{clip}((e-S)/R,0,1)$
- 学习率 StepLR：
  $$ \mathrm{lr}(e) = \mathrm{lr}_0\cdot \gamma^{\lfloor e/\mathrm{step\_size}\rfloor},\quad 0<\gamma<1 $$
- 优化器：Adam（β=(0.9,0.999)，可设 weight_decay），检测头参数可用 `--jam-head-lr-mult` 提升学习率。

---

## A.5 指标与日志字段

- 训练打印/日志写入（每 epoch）：
  - `train_total`：训练总损失（含掩码项）
  - `train_mask`：训练掩码 BCE（仅 `--use-jammer`）
  - `val_total`：验证总损失
  - `val_nmse`：验证 NMSE（主任务）
  - `val_mask`：验证掩码 BCE（仅 `--use-jammer`）
  - `val_clean`：无干扰评估 NMSE（开启 `--eval-clean` 时）
  - `gate`：当前门控强度 g(e)
  - `lambda_mask`：当前权重 λ_mask(e)

---

## A.6 主要超参数与 CLI 映射（`train.py`）

- 数据：`--train-r-path`（历史），`--train-t-path`（未来），`--is-u2d`（FDD 键选择）
- 训练：`--epochs`，`--batch-size`，`--lr`，`--lr-step-size`，`--lr-gamma`，`--weight-decay`，`--num-workers`
- 设备：`--multi-gpu`，`--device-ids`，`--gpu-id`
- 预训练/保存：`--pretrained-path`，`--save-path`，`--save-every`，`--save-last`
- 干扰：`--use-jammer`，`--jammer-cfg`（JSON，可含 `num_subcarriers=K`）
- 掩码/门控：
  - `--lambda-mask`，`--lambda-mask-final`，`--lambda-mask-switch`，`--lambda-mask-ramp`
  - `--jam-gate-start`，`--jam-gate-strength`，`--jam-gate-warmup`
  - `--jam-head-hidden-ratio`，`--jam-head-lr-mult`
- 评估与日志：`--eval-clean`，`--log-file`

---

## A.7 复杂度与开销（检测头）

- 参量量级：$\approx 2\,D\,(rD) + O(D) = O(r\,D^2)$，远小于 GPT 主干（千万级）。
- 计算量：两次矩阵乘，线性随 B、L、D 增长。
- 工程优势：
  - 轻量、可解释、位置级概率输出；
  - 可通过 g 实现按场景可控（干净场景 g=0）。

---

## A.8 经验推荐与对照配置

- 保守门控：`g_target∈[0.3, 0.6]`，warmup 10–20 epoch；
- λ_mask：`λ_F∈[0.2, 0.6]`，switch 5–30，ramp 15–40；
- 干扰强度（示例 JSON）：
  - `jammer_very_light.json`：`jam_prob≈0.25`，`JSR≤10 dB`，较小带宽/占空；
  - `jammer_light.json`：`jam_prob≈0.4`，`JSR≤12 dB`；
  - `jammer_moderate_soft.json`：`jam_prob≈0.6`，`JSR≤20 dB`，中等偏重。

---

## A.9 设计动机与与注意力的关系（简述）

- 检测头以位置级 MLP 输出 $M\in[0,1]$，与监督掩码直接对齐（BCE），以乘性门控实现“检测即去扰”。
- 与注意力对比：
  - 复杂度与显存更低；概率掩码更易阈值化与可视化；
  - 与主干 GPT 的全局建模解耦，便于用 (g, λ) 做可控 trade-off；
  - 推理时可快速禁用（g=0）适配干净场景。

---

### 附：关键公式（便于论文引用）

- 掩码：$ M = \sigma(\mathrm{MLP}(X)) $
- 门控：$ \tilde{X} = X \odot (1 - g\,M) $
- NMSE：$ \mathrm{NMSE} = \frac{\|\hat{Y}-Y\|_2^2}{\|Y\|_2^2 + \varepsilon} $
- 掩码 BCE：$ L_{mask} = -\,\mathrm{mean}[\, m\log p + (1-m)\log(1-p)\,] $
- 总损失：$ L_{total} = L_{pred} + \lambda_{mask}\, L_{mask} $
- g 调度：$ g(e) = g_{start} + (g_{target}-g_{start})\,\mathrm{clip}(e/W,0,1) $
- λ 调度：$ \lambda(e) = \lambda_0 + (\lambda_F-\lambda_0)\,\mathrm{clip}((e-S)/R,0,1) $
- 干扰强度：$ \sigma_{jam} = \sqrt{\mathrm{mean}(|H|^2)\,10^{\mathrm{JSR}/10}/2} $

---

参考文件：`data.py`（数据/增强）、`models/GPT4CP.py`（模型）、`train.py`（训练与调度）、`configs/*.json`（干扰配置）。
