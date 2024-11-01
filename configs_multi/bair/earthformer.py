method = 'MultiEarthformer'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 64
hid_T = 256
N_T = 8
N_S = 2
# training
lr = 1e-3
batch_size = 1
drop_path = 0.1
sched = 'onecycle'
initial_shape = [4, 64, 64, 3]
input_shape = [4, 32, 32, 3]    # backbone shape
in_shape = [4, 3, 64, 64]
metrics = ['mse', 'mae', 'ssim', 'psnr']
pre_seq_length = 2
aft_seq_length = 12
total_length = 16