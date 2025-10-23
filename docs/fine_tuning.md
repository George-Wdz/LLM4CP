# Jammer-Aware Fine-Tuning Guide

This repository now supports a jammer-aware fine-tuning routine that augments the historical channel observations with synthetic interference, trains a lightweight detection head to predict jammer masks, and gates the GPT-based predictor on the fly. The workflow keeps the original LLM4CP weights intact while adding the following components:

- **Jammer augmentation** (enabled via `--use-jammer`) injects wideband, partial-band, multi-tone, pulsed, and frequency-hopping interference with JSR sampled from **[-10, +20] dB** and surfaces the ground-truth mask to the loader.
- **Mask head + gating** adds a tiny MLP (`jam_head`) in `models/GPT4CP.py` that predicts a soft mask over each token; tokens with high jammer probability are down-weighted before entering GPT-2.
- **Joint loss** keeps the original NMSE objective while adding a BCE penalty between predicted and ground-truth masks (`--lambda-mask`).

The procedure is a **full-model fine-tune**: GPT-2 parameters remain trainable but you can optionally freeze selected layers or use parameter groups to lower their learning rate. This approach is preferred here because we have a modest model size and strong supervision (mask + prediction). If VRAM is constrained you can plug in PEFT/LoRA later—see the final section for tips.

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data and weights**
   - Place the provided `.mat` datasets under `Dataset/train_data` and `Dataset/test_data` (matching the README instructions).
   - Download the original LLM4CP checkpoints into `Weights/`.

3. **Launch fine-tuning**

   ```bash
   python train.py \
     --train-r-path Dataset/train_data/H_U_his_train.mat \
     --train-t-path Dataset/train_data/H_U_pre_train.mat \
     --pretrained-path Weights/U2U_LLM4CP.pth \
     --save-path Weights/U2U_LLM4CP_jam.pth \
     --use-jammer \
     --lambda-mask 1.0 \
     --jam-gate-strength 1.0 \
     --jam-head-hidden-ratio 0.5 \
     --epochs 100 \
     --batch-size 256
   ```

   The script dynamically synthesizes interference per batch; no new `.mat` files are needed. Validation uses the same augmentation so you can monitor robustness as training progresses.

4. **Evaluate** using the existing `test_tdd_full.py` / `test_fdd_full.py` scripts after pointing them to the new checkpoint. For jammer stress tests, extend those scripts with the same augmentation helpers and sweep JSR levels.

## Configuration knobs

- **Jammer config**: pass a JSON file via `--jammer-cfg` to override ranges (bandwidth fraction, hop length, tone count, etc.). See `DEFAULT_JAMMER_CFG` in `data.py` for available keys.
- **Mask weighting (`--lambda-mask`)**: start at 1.0; raise if detection lags behind prediction.
- **Gating strength (`--jam-gate-strength`)**: scales how aggressively masked tokens are suppressed (1.0 = zero out, 0.5 = halve amplitude).
- **Hidden ratio (`--jam-head-hidden-ratio`)**: shrinks or enlarges the MLP behind the jam head; values in [0.25, 0.75] work well.

## Fine-tuning vs. PEFT/LoRA

This update keeps all GPT-2 parameters trainable for maximal capacity. If you need a lighter footprint:

- **Freeze GPT-2**: add `for param in model.gpt2.parameters(): param.requires_grad = False` before constructing the optimizer to update only the residual CNN stacks + jam head.
- **LoRA/PEFT**: integrate [`peft`](https://github.com/huggingface/peft) (already listed in `requirements.txt`) by wrapping `model.gpt2` with a LoRA adapter. You will still use the same dataset/loader but only a fraction of parameters will update.

Both strategies reuse the same `--use-jammer` pipeline, so you can iterate quickly between full fine-tuning and parameter-efficient variants depending on hardware and project goals.
