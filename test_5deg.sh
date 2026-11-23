#!/bin/bash

### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# micromamba environment
source ~/.bashrc
conda activate seoir

# Project directory
cd /project/train/src_repo/SEOIR/

### ===== [2] Configurations =====
DENOISE_DIR="data_dir/data/test/denoise/"
DERAIN_DIR="data_dir/data/test/derain/"
DEHAZE_DIR="data_dir/data/test/dehaze/"
GOPRO_DIR="data_dir/data/test/deblur/"
ENHANCE_DIR="data_dir/data/test/enhance/"

### ===== [3] Launch Training =====
python test.py \
    --trainset SEOIR \
    --mode 6 \
    --denoise_path "$DENOISE_DIR" \
    --derain_path "$DERAIN_DIR" \
    --dehaze_path "$DEHAZE_DIR" \
    --gopro_path "$GOPRO_DIR" \
    --enhance_path "$ENHANCE_DIR" \
    --ckpt_name 5deg/seoir-AiO5D.ckpt \
    --output_path ./outputs/5deg