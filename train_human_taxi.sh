#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -p vip_gpu_ailab
#SBATCH -A ai4bio

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /home/bingxing2/ailab/suencheng/encheng/code/OpenSTL/tools/train.py --datanames human taxibj --lr 1e-3 --configs configs_multi/human/SimVP.py configs_multi/taxibj/SimVP.py --ex_name taxi_human_simvp

sleep 360000
