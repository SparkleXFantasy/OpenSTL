CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py \
 --datanames human taxibj bair \
 -c configs_multi/human/earthformer.py\
 --lr 1e-4 \
 --configs configs_multi/human/earthformer.py configs_multi/taxibj/earthformer.py configs_multi/bair/earthformer.py \
 --ex_name earthformer_taxi_human_bair \
 --data_root /home/data1/songxiufeng/st_world_model \