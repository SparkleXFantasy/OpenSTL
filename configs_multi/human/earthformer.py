method = 'MultiEarthformer'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
batch_size = 1
sched = 'cosine'
warmup_epoch = 0
initial_shape = [4, 256, 256, 3]
in_shape = [4, 3, 256, 256]
input_shape = [4, 32, 32, 3]    # backbone shape
pre_seq_length = 4
aft_seq_length = 4
total_length = 8
metrics = ['mse', 'mae', 'ssim', 'psnr']
drop_path = 0.1