import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1

def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p

def MAE(pred, true, spatial_norm=False, batch_size=256):
    # 初始化 MAE 和批次数
    mae = 0.0
    total_samples = pred.shape[0]
    num_batches = total_samples // batch_size + (1 if total_samples % batch_size != 0 else 0)

    for i in range(num_batches):
        # 获取当前批次的预测值和真实值
        batch_pred = pred[i * batch_size:(i + 1) * batch_size]
        batch_true = true[i * batch_size:(i + 1) * batch_size]
        
        # 计算当前批次的 MAE
        if not spatial_norm:
            batch_mae = np.mean(np.abs(batch_pred - batch_true), axis=(0, 1)).sum()
        else:
            norm = batch_pred.shape[-1] * batch_pred.shape[-2] * batch_pred.shape[-3]
            batch_mae = np.mean(np.abs(batch_pred - batch_true) / norm, axis=(0, 1)).sum()

        # 累加当前批次的 MAE
        mae += batch_mae

    # 计算最终的平均 MAE
    mae /= num_batches
    return mae
# def MAE(pred, true, spatial_norm=False):
#     if not spatial_norm:
#         return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
#     else:
#         norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
#         return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()

def MSE(pred, true, spatial_norm=False, batch_size=256):
    # 初始化 MSE 和批次数
    mse = 0.0
    total_samples = pred.shape[0]
    num_batches = total_samples // batch_size + (1 if total_samples % batch_size != 0 else 0)

    for i in range(num_batches):
        # 获取当前批次的预测值和真实值
        batch_pred = pred[i * batch_size:(i + 1) * batch_size]
        batch_true = true[i * batch_size:(i + 1) * batch_size]
        
        # 计算当前批次的 MSE
        if not spatial_norm:
            batch_mse = np.mean((batch_pred - batch_true) ** 2, axis=(0, 1)).sum()
        else:
            norm = batch_pred.shape[-1] * batch_pred.shape[-2] * batch_pred.shape[-3]
            batch_mse = np.mean((batch_pred - batch_true) ** 2 / norm, axis=(0, 1)).sum()

        # 累加当前批次的 MSE
        mse += batch_mse

    # 计算最终的平均 MSE
    mse /= num_batches
    return mse
# def MSE(pred, true, spatial_norm=False):
#     if not spatial_norm:
#         return np.mean((pred-true)**2, axis=(0, 1)).sum()
#     else:
#         norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
#         return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def POD(hits, misses, eps=1e-6):
    """
    probability_of_detection
    Inputs:
    Outputs:
        pod = hits / (hits + misses) averaged over the T channels
        
    """
    pod = (hits + eps) / (hits + misses + eps)
    return np.mean(pod)

def SUCR(hits, fas, eps=1e-6):
    """
    success_rate
    Inputs:
    Outputs:
        sucr = hits / (hits + false_alarms) averaged over the D channels
    """
    sucr = (hits + eps) / (hits + fas + eps)
    return np.mean(sucr)

def CSI(hits, fas, misses, eps=1e-6):
    """
    critical_success_index 
    Inputs:
    Outputs:
        csi = hits / (hits + false_alarms + misses) averaged over the D channels
    """
    csi = (hits + eps) / (hits + misses + fas + eps)
    return np.mean(csi)

def sevir_metrics(pred, true, threshold):
    """
    calcaulate t, p, hits, fas, misses
    Inputs:
    pred: [N, T, C, L, L]
    true: [N, T, C, L, L]
    threshold: float
    """
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    pred = pred.astype(np.float16)
    true = true.astype(np.float16)
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean

    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None
    
    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses) 
        
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                # 将 (C, H, W) 转换为 (H, W, C)
                img_norm = pred[b, f].transpose(1, 2, 0)
                true_img = true[b, f].transpose(1, 2, 0)

                # 计算数据范围
                data_range = img_norm.max() - img_norm.min()

                # 打印调试信息以确认图像形状和数据范围
                #print(f"Image shape: {img_norm.shape}, Win size: 3, Data range: {data_range}")

                # 扩展双通道为三通道
                if img_norm.shape[-1] == 2:
                    img_norm = np.concatenate([img_norm, np.zeros((img_norm.shape[0], img_norm.shape[1], 1))], axis=-1)
                    true_img = np.concatenate([true_img, np.zeros((true_img.shape[0], true_img.shape[1], 1))], axis=-1)
                    #print(f"Extended image to three channels: {img_norm.shape}")

                # 设置 multichannel 参数
                if img_norm.shape[-1] == 1:
                    multichannel = False  # 单通道灰度图像
                else:
                    multichannel = True   # 多通道图像

                # 尝试计算 SSIM
                try:
                    ssim += cal_ssim(img_norm,
                                    true_img, 
                                    win_size=3, 
                                    data_range=data_range, 
                                    multichannel=multichannel)
                except ValueError as e:
                    print(f"Error calculating SSIM for image of size {img_norm.shape}: {e}")
                    continue

        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])



    # if 'ssim' in metrics:
    #     ssim = 0
    #     for b in range(pred.shape[0]):
    #         for f in range(pred.shape[1]):
    #             img_norm = pred[b, f].transpose(0, 2)
    #             data_range = img_norm.max() - img_norm.min()
    #             #print(f"Image shape: {img_norm.shape}")

    #             ssim += cal_ssim(img_norm,
    #                              true[b, f].transpose(0, 2), multichannel=True, win_size=3, data_range=data_range)
    #     eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred = pred.transpose(0, 1, 3, 4, 2)
        true = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                lpips += cal_lpips(pred[b, f], true[b, f])
        eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log
