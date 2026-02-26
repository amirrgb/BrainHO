# Learning the Hierarchical Organization in Brain Network for Brain Disorder Diagnosis

This repository contains the official PyTorch implementation of **BrainHO**.

## Installation

Create a virtual environment and install the required dependencies:

```bash
conda create -n BrainHO python==3.9.0
```
```bash
conda activate BrainHO
```
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install -r requirements.txt
```

## Data Preparation

Please download the required fMRI datasets directly from the official ABIDE and REST-meta-MDD websites.

### preprocess the ABIDE dataset
We directly utilized the open-source preprocessed data provided by the Brain Network Transformer (BNT).

### preprocess the REST-meta-mdd dataset
```bash
python fetch_rest_meta_mdd_cc200.py
```


## Training & Evaluation

### BrainHO
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










