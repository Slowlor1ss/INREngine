"""
Small numpy helpers for the diff view
"""
import numpy as np


def compute_diff_heatmap(current: np.ndarray, target: np.ndarray, tolerance: float) -> np.ndarray:
    if current.shape != target.shape:
        raise ValueError(f"compute_diff_heatmap: shape mismatch {current.shape} vs {target.shape} "
                          f"-- resize one to match the other before calling this.")

    diff = np.abs(current.astype(np.float32) - target.astype(np.float32))
    magnitude = diff.max(axis=-1)  # worst channel per pixel, in [0,1]

    severity = np.clip((magnitude - tolerance) / max(1e-6, 1.0 - tolerance), 0.0, 1.0)

    heatmap = np.zeros_like(current, dtype=np.float32)
    heatmap[..., 0] = severity          # red ramps with severity
    heatmap[..., 1] = severity * 0.35   # a little green mixed in -> orange/yellow at high severity
    heatmap[magnitude <= tolerance] = 0.0
    return heatmap


def nearest_resize(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Cheap nearest-neighbour resize so diff mode still works when the live
    render resolution doesn't match the target's native resolution """
    h, w, _ = arr.shape
    ys = (np.arange(new_h) * h / new_h).astype(np.int32)
    xs = (np.arange(new_w) * w / new_w).astype(np.int32)
    return arr[ys][:, xs]
