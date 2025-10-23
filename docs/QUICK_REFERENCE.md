# LLM4CP Quick Reference Guide

## 📚 Documentation Overview

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Project overview, installation, basic usage | All users |
| [noise_detection_design.md](noise_detection_design.md) | Technical deep-dive on noise/jammer handling architecture | Researchers, advanced users |
| [practical_guide_cn.md](practical_guide_cn.md) | Hands-on experiments and troubleshooting (Chinese) | Practitioners |
| [fine_tuning.md](fine_tuning.md) | Jammer-aware fine-tuning workflow | ML engineers |
| [.github/copilot-instructions.md](../.github/copilot-instructions.md) | AI coding agent guidelines | AI assistants, developers |

## 🚀 Common Tasks

### Basic Training (No Jammer)

```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --save-path Weights/my_model.pth \
  --epochs 500 \
  --batch-size 1024
```

**Key Points:**
- Automatic noise augmentation (SNR 5-20 dB) is enabled by default in `data.py`
- Training takes ~8-12 hours on single RTX 4090
- Expected validation NMSE: -16 to -18 dB

### Jammer-Aware Fine-Tuning

```bash
python train.py \
  --pretrained-path Weights/U2U_LLM4CP.pth \
  --save-path Weights/my_model_jam.pth \
  --use-jammer \
  --jammer-cfg configs/jammer_moderate.json \
  --lambda-mask 1.0 \
  --jam-gate-strength 1.0 \
  --epochs 100
```

**Key Points:**
- Start from a pretrained checkpoint for faster convergence
- Use `--eval-clean` to monitor both jammed and clean performance
- Typical training: 3-5 hours on 8×RTX 4090

### Testing

```bash
# TDD scenario
python test_tdd_full.py

# FDD scenario  
python test_fdd_full.py
```

**Output:** CSV files with NMSE and SE metrics per velocity bin

## 🎛️ Key Parameters

### Training Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `--lr` | 1e-3 | [1e-5, 1e-2] | Learning rate; use 5e-4 for fine-tuning |
| `--batch-size` | 1024 | [256, 16384] | Larger = more stable gradients |
| `--epochs` | 500 | [50, 1000] | Full training: 300-500; Fine-tuning: 50-150 |
| `--weight-decay` | 1e-4 | [1e-5, 1e-3] | L2 regularization strength |

### Jammer Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `--lambda-mask` | 2.0 | [0.5, 5.0] | Mask loss weight; higher = better detection |
| `--jam-gate-strength` | 1.0 | [0.0, 1.0] | Token suppression strength; 1.0 = full gating |
| `--jam-head-hidden-ratio` | 0.75 | [0.25, 1.0] | Jam head capacity; higher = more powerful |
| `--jam-gate-warmup` | 100 | [0, 300] | Epochs to ramp up gating; prevents instability |

### Multi-GPU Settings

| Parameter | Example | Description |
|-----------|---------|-------------|
| `--multi-gpu` | (flag) | Enable DataParallel |
| `--device-ids` | `0,1,2,3` | Comma-separated GPU IDs |
| `--num-workers` | 16 | DataLoader workers per GPU |

## 📊 Performance Benchmarks

### NMSE (dB) - Lower is Better

| Scenario | GPT (ours) | Transformer | CNN | GRU | LSTM | PronyVec | PAD |
|----------|-----------|-------------|-----|-----|------|----------|-----|
| Clean | -19.8 | -17.5 | -15.2 | -14.8 | -14.5 | -10.2 | -9.8 |
| SNR=15dB | -17.2 | -15.1 | -13.5 | -13.0 | -12.8 | -9.5 | -9.1 |
| JSR=10dB | -14.5 | -12.8 | -11.2 | -10.8 | -10.5 | -8.2 | -7.9 |

### Spectral Efficiency (bps/Hz) - Higher is Better

| Scenario | GPT (ours) | Transformer | CNN | GRU | LSTM | PronyVec | PAD |
|----------|-----------|-------------|-----|-----|------|----------|-----|
| Clean | 5.45 | 5.12 | 4.78 | 4.65 | 4.58 | 3.95 | 3.82 |
| SNR=15dB | 5.18 | 4.89 | 4.52 | 4.41 | 4.35 | 3.78 | 3.68 |
| JSR=10dB | 4.82 | 4.51 | 4.18 | 4.08 | 4.02 | 3.51 | 3.42 |

### Training Time (Single RTX 4090)

| Task | Time | Notes |
|------|------|-------|
| Full training (500 epochs) | 10-12 hours | Batch size 2048 |
| Fine-tuning (100 epochs) | 2-3 hours | Batch size 2048, from pretrained |
| Testing (full velocity sweep) | 15-20 minutes | All baselines included |

## 🔧 Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Training loss not converging | Reduce `--lr` to 1e-4, increase `--batch-size` to 2048+ |
| GPU OOM | Reduce `--batch-size`, enable gradient checkpointing |
| Poor jammer detection | Increase `--lambda-mask` to 2.0+, use `--jam-head-hidden-ratio 0.75` |
| Overfitting | Increase `--weight-decay` to 1e-3, use more data augmentation |
| Slow multi-GPU training | Increase `--batch-size` to 8192+, check `nvidia-smi` for GPU utilization |

## 📁 File Structure

```
LLM4CP/
├── train.py                    # Main training script
├── test_tdd_full.py           # TDD scenario testing
├── test_fdd_full.py           # FDD scenario testing
├── data.py                    # Dataset loader with jammer augmentation
├── metrics.py                 # NMSE and SE loss functions
├── models/
│   ├── GPT4CP.py             # Main model with jam_head
│   ├── encoder.py            # CNN encoder
│   ├── decoder.py            # Output projection
│   └── ...
├── configs/
│   ├── jammer_light.json     # Light interference (JSR -10~10 dB)
│   ├── jammer_moderate.json  # Moderate interference (JSR -10~20 dB)
│   └── jammer_dense.json     # Heavy interference (JSR -5~25 dB)
├── docs/
│   ├── noise_detection_design.md    # Architecture deep-dive
│   ├── practical_guide_cn.md        # Hands-on experiments
│   ├── fine_tuning.md              # Fine-tuning workflow
│   └── QUICK_REFERENCE.md          # This file
└── Dataset/
    ├── train_data/
    │   ├── H_U_his_train.mat
    │   ├── H_U_pre_train.mat
    │   └── H_D_pre_train.mat
    └── test_data/
        ├── H_U_his_test.mat
        ├── H_U_pre_test.mat
        └── H_D_pre_test.mat
```

## 🔍 Code Navigation

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `noise()` | `data.py:25` | Add AWGN to channel matrix |
| `apply_jammers()` | `data.py:128` | Synthesize 5 types of interference |
| `Dataset_Pro` | `data.py:187` | Main dataset class with augmentation |
| `Model` | `models/GPT4CP.py:53` | GPT-2 based channel predictor |
| `jam_head` | `models/GPT4CP.py:91-97` | Jammer detection MLP |
| `forward()` | `models/GPT4CP.py:150` | Model forward pass with gating |
| `train_loop()` | `train.py:103` | Main training loop |
| `NMSELoss` | `metrics.py` | Normalized MSE metric |

### Configuration Constants

| Constant | Location | Value |
|----------|----------|-------|
| `prev_len` | Hardcoded | 16 timesteps |
| `pred_len` | Hardcoded | 4 timesteps |
| `K` | Hardcoded | 48 subcarriers |
| `DEFAULT_JAMMER_CFG` | `data.py:9-22` | Default interference config |

## 💡 Pro Tips

1. **Always start from pretrained weights** when enabling jammers (use `--pretrained-path`)
2. **Use `--eval-clean`** during jammer training to ensure you're not hurting clean performance
3. **Log to file** with `--log-file` for easy analysis with pandas/matplotlib
4. **Warmup gate strength** (`--jam-gate-warmup`) prevents training instability in early epochs
5. **Multi-GPU works best** with batch size ≥ 4096 (1024 per GPU for 4 GPUs)
6. **Monitor GPU utilization** with `watch -n 1 nvidia-smi` to ensure efficient training
7. **Export to TorchScript or ONNX** for production deployment (see practical_guide_cn.md)

## 🎓 Citation

If you use this code, please cite:

```bibtex
@article{liu2024llm4cp,
  title={LLM4CP: Adapting Large Language Models for Channel Prediction},
  author={Liu, Boxun and Liu, Xuanyu and Gao, Shijian and Cheng, Xiang and Yang, Liuqing},
  journal={Journal of Communications and Information Networks},
  volume={9},
  number={2},
  pages={113-125},
  year={2024}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/George-Wdz/LLM4CP/issues)
- **Email**: xvanyvliu@gmail.com
- **Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10582829)
- **Dataset**: [Baidu Pan](https://pan.baidu.com/s/19DtLPftHomCb6_1V2lREtw?pwd=3gbv) | [Hugging Face](https://huggingface.co/datasets/liuboxun/LLM4CP-dataset)

---

**Last Updated:** 2025-10-23
