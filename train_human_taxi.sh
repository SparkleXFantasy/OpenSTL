#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p vip_gpu_ailab
#SBATCH -A ai4bio

conda activate OpenSTL
cd encheng/code/OpenSTL
module load cudnn/8.2.1.32_cuda11.x

module load cuda/11.7

python tools/train.py --datanames human taxibj bair --lr 1e-4 --configs configs_multi/human/SimVP.py configs_multi/taxibj/SimVP.py configs_multi/bair/SimVP.py --ex_name simvp_taxi_human_bair
