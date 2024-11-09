method = 'MultiSimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'
hid_S = 64
hid_T = 256
N_T = 8
N_S = 4
# training
lr = 1e-4
batch_size = 16
drop_path = 0.1
sched = 'onecycle'
warmup_epoch = 0
metric_threshold = 74
in_shape = [2, 3, 128, 128]
pre_seq_length = 2
aft_seq_length = 5
total_length = 8
metrics = ['mse', 'mae', 'ssim', 'psnr']
drop_path = 0.1