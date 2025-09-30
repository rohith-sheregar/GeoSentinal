# risk.py
# Slope and risk computation functions.

import numpy as np

def norm(a):
    a = np.array(a, dtype=float)
    mn = a.min()
    mx = a.max()
    if mx - mn < 1e-9:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def compute_slope(dem, dx=1.0, dy=1.0):
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.hypot(gx, gy)
    return slope

def compute_risk(dem, rainfall=0.0, ndvi=None, weights=(0.5, 0.35, 0.15)):
    s = norm(compute_slope(dem))
    r = norm(np.full_like(s, rainfall))
    v = norm(ndvi) if ndvi is not None else np.zeros_like(s)
    w_s, w_r, w_v = weights
    risk = w_s * s + w_r * r + w_v * (1 - v)
    return risk
