#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p vip_gpu_ailab
#SBATCH -A ai4bio
export PYTHONPATH=/home/bingxing2/ailab/suencheng/encheng/simvp_5/OpenSTL:$PYTHONPATH
module load cudnn/8.2.1.32_cuda11.x
module load cuda/11.7
cd encheng/simvp_5/OpenSTL
source activate OpenSTL

python tools/train.py --datanames human taxibj bair city sevir --lr 1e-4 --configs configs_multi/human/SimVP.py configs_multi/taxibj/SimVP.py configs_multi/bair/SimVP.py configs_multi/city/SimVP.py configs_multi/sevir/SimVP.py --ex_name simvp_taxi_human_bair_city_sevir