method = 'MultiSimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None
hid_S = 32
hid_T = 128
N_T = 8
N_S = 2
# training
lr = 1e-3
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0
in_shape = [4, 2, 32, 32]
pre_seq_length = 4
aft_seq_length = 4
total_length = 8
metrics = ['mse', 'mae', 'ssim', 'psnr']