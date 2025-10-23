# LLM4CP
B. Liu, X. Liu, S. Gao, X. Cheng and L. Yang, "LLM4CP: Adapting Large Language Models for Channel Prediction," in Journal of Communications and Information Networks, vol. 9, no. 2, pp. 113-125, June 2024, doi: 10.23919/JCIN.2024.10582829. [[paper]](https://ieeexplore.ieee.org/document/10582829)
<br>

## Dependencies and Installation
- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU + CUDA
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
The datasets used in this paper can be downloaded in the following links.  
[[Training Dataset]](https://pan.baidu.com/s/19DtLPftHomCb6_1V2lREtw?pwd=3gbv)
[[Testing Dataset]](https://pan.baidu.com/s/10KzmwC1jncozOGNZ02Hlaw?pwd=sxfd)
👉 Alternative Huggingface download link: [Dataset](https://huggingface.co/datasets/liuboxun/LLM4CP-dataset)

## Dataset Generation
We generate dataset via [QuaDRiGa](https://quadriga-channel-model.de/). To assist researchers in the field of channel prediction, we have provided a runnable demo file in the `data_generation` folder. For more detailed information about the QuDRiGa generator, please refer to its user documentation `uadriga_documentation_v2.8.1-0.pdf`.


## Get Started
Training and testing codes are in the current folder. 

-   The code for training is in `train.py`, while the code for test is in `test_tdd_full.py` and `test_fdd_full.py`. we also provide our pretrained model in [[Weights]](https://pan.baidu.com/s/1lysOqCyw44SGDQrH33Os5Q?pwd=nmqw).
👉 Alternative Huggingface download link:[Model weights](https://huggingface.co/liuboxun/LLM4CP).
    
-   For full shot training, you need to set the file_path in the main function to match your training dataset. For example, if you want to try a full-shot experiment in a TDD scenario, you need to modify the `train_TDD_r_path` and `train_TDD_t_path` in `train.py` to the locations of your downloaded `H_U_his_train.mat` and `H_U_pre_train.mat`, respectively. Then, you can run `train.py`.
-   For few shot training, you need to set the file_path in the main function to match your training dataset. Then, you can set `is_few=1` when creating the training set in `train.py` like this: `train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_few=1)` and run `train.py`.

-   For testing, you also need to set the file_path in the main function to match your testing dataset. Then, you can run `test_tdd_full.py` to obtain the results in Figure 7 of the paper, and you can run `test_fdd_full.py` to obtain the results in Figure 8 of the paper. You can also try loading the data under `Testing Dataset/Umi` to test the models' zero-shot performance.

## Jammer-aware Fine-Tuning Workflow

Use the enhanced pipeline (documented in `docs/fine_tuning.md`) to fine-tune LLM4CP with synthetic interference and the jam-mask head introduced in this repo update.

1. **Install dependencies**

  ```bash
  conda activate llm4cp_env  # or create your own env
  pip install -r requirements.txt
  ```

1. **Prepare datasets and checkpoints**

  ```text
  Dataset/train_data/H_U_his_train.mat
  Dataset/train_data/H_U_pre_train.mat
  Dataset/test_data/H_U_his_test.mat
  Dataset/test_data/H_U_pre_test.mat
  Weights/U2U_LLM4CP.pth  # original checkpoint provided by authors
  ```

1. **(Optional but recommended) Cache GPT-2 weights locally** – avoids repeated downloads from Hugging Face, useful on restricted networks.

  ```bash
  mkdir -p hf_models/gpt2
  for file in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json pytorch_model.bin; do
    wget -O hf_models/gpt2/$file https://hf-mirror.com/gpt2/resolve/main/$file
  done
  export GPT2_LOCAL_PATH=$(pwd)/hf_models/gpt2
  export TRANSFORMERS_OFFLINE=1
  ```

1. **Launch fine-tuning (example: 8×RTX 4090)**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --train-r-path ../Dataset/train_data/H_U_his_train.mat \
    --train-t-path ../Dataset/train_data/H_U_pre_train.mat \
    --pretrained-path ../Weights/U2U_LLM4CP.pth \
    --save-path ../Weights/U2U_LLM4CP_jam.pth \
    --use-jammer \
    --lambda-mask 1.0 \
    --jam-gate-strength 1.0 \
    --jam-head-hidden-ratio 0.5 \
    --epochs 100 \
    --batch-size 2048 \
    --multi-gpu \
    --device-ids 0,1,2,3,4,5,6,7
  ```

  The loader synthesizes wideband, partial-band, multi-tone, pulsed, and frequency-hopping jammers with JSR sampled from **[-10, +20] dB**, and the model trains with NMSE + mask BCE losses.

1. **Evaluate** using the existing scripts (update dataset paths as needed):

  ```bash
  python test_tdd_full.py
  python test_fdd_full.py
  ```

  Extend these scripts with the same jammer configuration to produce NMSE/SE vs. JSR curves.

For additional knobs (custom jammer JSON configs, LoRA/PEFT tips, etc.) see `docs/fine_tuning.md`.

## Documentation

- **[Quick Reference](docs/QUICK_REFERENCE.md)**: Fast lookup guide with common commands, parameters, benchmarks, and troubleshooting
- **[Noise Detection and Channel Prediction Design](docs/noise_detection_design.md)** (Chinese & English): Comprehensive guide explaining the noise and interference handling strategy, U2D channel prediction principle, model architecture, and effectiveness analysis
- **[Practical Guide (Chinese)](docs/practical_guide_cn.md)**: Hands-on guide with experiments, performance tuning tips, troubleshooting, and best practices for training under noise and interference scenarios
- **[Fine-Tuning Guide](docs/fine_tuning.md)**: Detailed workflow for jammer-aware fine-tuning with parameter-efficient methods

## Citation

If you find this repo helpful, please cite our paper.

```latex
@article{liu2024llm4cp,
  title={LLM4CP: Adapting Large Language Models for Channel Prediction},
  author={Liu, Boxun and Liu, Xuanyu and Gao, Shijian and Cheng, Xiang and Yang, Liuqing},
  journal={arXiv preprint arXiv:2406.14440},
  year={2024}
```
