# LLM4CP Copilot Instructions

## Project Overview
LLM4CP (Large Language Model for Channel Prediction) adapts GPT-2 for OFDM channel prediction. It uses a hybrid architecture with a ResNet-like encoder for feature extraction and a GPT-2 backbone for sequence modeling.

## Architecture & Core Components
- **Model (`models/GPT4CP.py`)**:
  - **Structure**: `Res_block` (CNN encoder) -> `GPT2Model` (Backbone) -> Projection to output.
  - **Input**: Complex channel matrices reshaped to real tensors.
  - **Jammer Head**: Optional MLP (`jam_head`) for jammer mask prediction, gated by `jam_gate_strength`.
  - **GPT-2 Loading**: Robust offline loading via `_resolve_local_gpt2_dir`. Checks `GPT2_LOCAL_PATH` env var or `hf_models/gpt2`.
- **Data Pipeline (`data.py`)**:
  - **Source**: `.mat` files (MATLAB v7.3) containing `H_U_his_*`, `H_U_pre_*`.
  - **Processing**: `Dataset_Pro` handles reshaping, noise injection (`noise()`), and jammer simulation (`apply_jammers`).
  - **Jammers**: Implements Wideband (`wb`), Partial-band (`pb`), Tone, Pulse, and Frequency Hopping (`fh`) jammers. Configured via `DEFAULT_JAMMER_CFG`.
- **Training (`train.py`)**:
  - **Loss**: NMSE (Normalized Mean Squared Error) + Optional BCE for jammer mask.
  - **Optimization**: Adam optimizer. Separate LR for jammer head (`--jam-head-lr-mult`).
  - **Scheduling**: `StepLR` for learning rate decay.

## Critical Workflows

### Training
Run `train.py` with paths to `.mat` files.
```bash
python train.py \
  --train-r-path Dataset/train_data/H_U_his_train.mat \
  --train-t-path Dataset/train_data/H_U_pre_train.mat \
  --save-path Weights/my_model.pth \
  --device cuda:0 \
  --use-jammer --lambda-mask 1.0  # For jammer-aware training
```
- **Multi-GPU**: Use `--multi-gpu --device-ids 0,1,2`.
- **Few-Shot**: Add `--few-shot` flag.

### Testing
Use `test_tdd_full.py` or `test_fdd_full.py`.
```bash
python test_tdd_full.py \
  --prev-path Dataset/test/H_U_his_test.mat \
  --pred-path Dataset/test/H_U_pre_test.mat \
  --weights-gpt Weights/my_model.pth \
  --use-jammer
```
- **Note**: Test scripts iterate over velocity bins and save CSV results.

### Environment Setup
- **Hugging Face**: Uses `https://hf-mirror.com` by default.
- **Offline Mode**: Set `GPT2_LOCAL_PATH` to `hf_models/gpt2` to avoid downloads.

## Coding Conventions & Patterns
- **Tensor Shapes**:
  - **Raw**: `(Batch, Time, K*Antennas)` complex.
  - **Model Input**: `(Batch*32, Time, 2*K*Antennas/32)` real. (Reshaped in `LoadBatch_ofdm`).
  - **Sequence Lengths**: Default `prev_len=16`, `pred_len=4`.
  - **Subcarriers (K)**: Default `K=64` (configurable).
- **Normalization**: RMS normalization is critical. `sample = sample / rms`.
- **Configuration**:
  - Jammer configs are JSON files (e.g., `configs/jammer_light.json`).
  - `num_subcarriers` in config must match model `K`.
- **Device Handling**:
  - Prefer passing `device` string (e.g., `cuda:0`) over hardcoded `cuda`.
  - `train.py` uses `torch.nn.DataParallel` for multi-GPU.

## Common Pitfalls
- **Dimension Mismatch**: Ensure `K`, `UQh`, `UQv`, etc., match between training and inference.
- **Checkpoint Loading**: `load_pretrained` handles both `Model` and `DataParallel` state dicts.
- **Randomness**: `data.py` uses a custom `_ensure_rng` to handle seeding for jammers.

