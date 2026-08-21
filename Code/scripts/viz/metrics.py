import time
from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


@dataclass
class MetricsSample:
    timestamp: float
    epoch: int
    cost: float
    mse: float
    rmse: float
    mae: float
    max_err: float
    psnr: float
    ssim: float


# List of fields/colums in the history table and a graph in MetricsWindow (add field here, compute below)
METRIC_FIELDS = ["mse", "rmse", "mae", "max_err", "psnr", "ssim"]
METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "max_err": "Max Error",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
}


def compute_image_metrics(current: np.ndarray, target: np.ndarray, epoch: int, cost: float) -> MetricsSample:
    """Current, target: (H, W, C) arrays, same shape, values in [0, 1]"""
    diff = current.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))

    # Same near zero MSE fallback as in our C++, to avoid -10*log10(0) -> inf
    psnr = 100.0 if mse <= 1e-10 else float(sk_psnr(target, current, data_range=1.0))

    # gaussian_weights = true + sigma=1.5 is the standard 11x11 equivalent windowed SSIM from the original paper
    ssim = float(sk_ssim(target, current, channel_axis=-1, data_range=1.0,
                          gaussian_weights=True, sigma=1.5, use_sample_covariance=False))

    return MetricsSample(timestamp=time.time(), epoch=epoch, cost=cost,
                          mse=mse, rmse=rmse, mae=mae, max_err=max_err, psnr=psnr, ssim=ssim)
