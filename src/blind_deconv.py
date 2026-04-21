#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:27:29 2026

@author: dbp
"""

import numpy as np
from scipy.signal import convolve2d

# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.maximum(kernel, 0.0)
    s = kernel.sum()
    if s <= 0:
        kernel = np.zeros_like(kernel)
        center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        kernel[center] = 1.0
        return kernel
    return kernel / s


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return normalize_kernel(ker)


def circular_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return convolve2d(image, kernel, mode="same", boundary="wrap")


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return np.clip(image + sigma * rng.standard_normal(image.shape), 0.0, 1.0)



# ============================================================
# OPÉRATEURS LINÉAIRES MATRICIELS
# ============================================================

def build_kernel_matrix(kernel: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Construit K_k tel que vec(k * f) = K_k vec(f),
     où vec est l'opérateur de vectorisation colonne par colonne (idx = i + j*H)
    """
    
    n = H * W
    kh, kw = kernel.shape
    Kk = np.zeros((n, n), dtype=float)

    ch = kh // 2
    cw = kw // 2

    for i in range(H):
        for j in range(W):
            row = i + j * H
            for u in range(kh):
                for v in range(kw):
                    ii = (i - (u - ch)) % H
                    jj = (j - (v - cw)) % W
                    col = ii + jj * H
                    Kk[row, col] += kernel[u, v]

    return Kk

def build_image_matrix(image: np.ndarray, kernel_shape: tuple[int, int]) -> np.ndarray:
    """
    Construit K_f tel que vec(k * f) = K_f vec(k),
    où vec est l'opérateur de vectorisation colonne par colonne (idx = i + j*H)
    """
    
    H, W = image.shape
    kh, kw = kernel_shape
    n = H * W
    Kf = np.zeros((n, kh * kw), dtype=float)

    idx = 0
    for u in range(kh):
        for v in range(kw):
            basis = np.zeros((kh, kw), dtype=float)
            basis[u, v] = 1.0
            conv = circular_convolve2d(image, basis)
            Kf[:, idx] = conv.reshape(-1, order="F")
            idx += 1

    return Kf
