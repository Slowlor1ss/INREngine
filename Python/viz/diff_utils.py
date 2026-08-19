"""
Small numpy helpers for the diff view
"""
import numpy as np


def compute_diff_heatmap(current: np.ndarray, target: np.ndarray, tolerance: float) -> np.ndarray:
    if current.shape != target.shape:
        raise ValueError(f"compute_diff_heatmap: shape mismatch {current.shape} vs {target.shape} "
                          f"-- resize one to match the other before calling this (see nearest_resize).")

    diff = np.abs(current.astype(np.float32) - target.astype(np.float32))
    magnitude = diff.max(axis=-1)  # worst channel per pixel, in [0,1]

    # 0 at/under tolerance, ramps to 1 with difference
    # Mask of pixels that are out of bounds
    out_of_tol = magnitude > tolerance
    severity = np.zeros_like(magnitude)
    # For the bad pixels, map their severity starting from a high base value
    # so even small error are still clearly visible
    base_red = 0.5
    raw_severity = (magnitude[out_of_tol] - tolerance) / max(1e-6, 1.0 - tolerance)
    severity[out_of_tol] = base_red + (1.0 - base_red) * raw_severity
    severity = np.clip(severity, 0.0, 1.0)

    heatmap = np.zeros(current.shape[:2] + (3,), dtype=np.float32)
    heatmap[..., 0] = severity        # red ramps up with severity
    heatmap[..., 2] = 1.0 - severity  # blue ramps down, pure blue at severity 0
    return heatmap


def apply_ghost_overlay(diff_heatmap: np.ndarray, ghost_image: np.ndarray, opacity: float) -> np.ndarray:
    opacity = np.clip(opacity, 0.0, 1.0)
    return diff_heatmap * (1.0 - opacity) + ghost_image * opacity


def nearest_resize(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Cheap nearest-neighbour resize so diff mode still works when the live
    render resolution doesn't match the target's native resolution """
    h, w, _ = arr.shape
    ys = (np.arange(new_h) * h / new_h).astype(np.int32)
    xs = (np.arange(new_w) * w / new_w).astype(np.int32)
    return arr[ys][:, xs]
