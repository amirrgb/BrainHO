# paper name

This repository contains the official PyTorch implementation of **mymodel**.

## Installation

Create a virtual environment and install the required dependencies:

```bash
conda create -n mymodel python==3.9.0
```
```bash
conda activate mymodel
```
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install -r requirements.txt
```

## Data Preparation

Please download the required fMRI datasets directly from the official ABIDE and REST-meta-MDD websites.

## Training & Evaluation

### mymodel
```bash
python train_t.py --session current_SOTA \
                  --rootdir . \
                  --cuda 1 \
                  --dataset abide_cc200 \
                  --dataset_path /home/path/path/dataset/abide_cc200/abide.npy \
                  --data_aug \
                  --max_epochs 200 \
                  --w_loss1 1.3 \
                  --temperature 2.0 \
                  --target_weight 0.2 \
                  --center_step_div 4 \
                  --slope 0.001 \
                  --save_ckpt
```










