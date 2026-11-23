#!/bin/bash
#SBATCH --job-name=deg5_test
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --nodelist=gcp-eu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --time=01:00:00
#SBATCH --output=./joblogs/deg5_test_ep109.log
#SBATCH --error=./joblogs/deg5_test_ep109.error

### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# micromamba environment
# source ~/.bashrc
source /project/miniconda3/etc/profile.d/conda.sh
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
    --ckpt_name 20260916_5deg/epoch=104-step=521850.ckpt \
    --output_path ./outputs/visual_fig4_20260916_ep104