#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:05:50 2026

@author: dbp

"""

import numpy as np
import casadi as ca
from src.dwt import *
from src.blind_deconv import *
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from skimage.restoration import richardson_lucy


# ============================================================
# QUESTION 8 : DECONVOLUTION AVEUGLE AVEC CASADI / IPOPT
# ============================================================

def solve_blind_joint_casadi(
    observed: np.ndarray,
    kernel_shape: tuple[int, int] = (5, 5),
    init_kernel: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Résout le problème 
        min_{f,k} 1/2 ||K(k) f - y||^2
        s.c. k >= 0, sum(k)=1, et 0 <= f <= 1
    """

    # TO DO
    f_est = observed
    k_est = np.zeros(kernel_shape)
    
    return f_est, k_est


# ================================================================
# QUESTION 9 : DECONVOLUTION AVEUGLE AVEC MINIMISATION ALTERNEE
# ================================================================


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projection euclidienne sur {x >= 0, sum(x)=1}.
    """
    
    # TO DO
    
    return v


def solve_ADM(
    observed: np.ndarray,
    kernel_shape: tuple[int, int] = (5, 5),
    lam: float = 2e-3,
    wavename: str = "db4",
    outer_iter: int = 100,
    init_kernel: np.ndarray | None = None,
    verbose: bool = True,
):
    """
    Minimisation alternée :
      - étape image : gradient projeté
      - étape noyau : gradient projeté sur le simplexe
    """
    
    # TO DO
    image_est = observed
    kernel_est = np.zeros(kernel_shape)
    

    return image_est, kernel_est


# ============================================================
# QUESTION 10 BLIND RICHARDSON-LUCY
# ============================================================

def blind_richardson_lucy(
    observed: np.ndarray,
    kernel_shape: tuple[int, int] = (5, 5),
    outer_iter: int = 30,
    init_image: np.ndarray | None = None,
    init_kernel: np.ndarray | None = None,
    verbose: bool = True,
):
    """
    Blind RL alterné :
      - update image via RL
      - update kernel via RL
      - normalisation du noyau

    observed      : image observée
    kernel_shape  : taille du noyau inconnu
    outer_iter    : nb d'itérations externes
    image_iter_per_outer : nb de mises à jour RL sur l'image
    kernel_iter_per_outer: nb de mises à jour RL sur le noyau
    """
    
    # TO DO
    image_est = observed
    kernel_est = np.zeros(kernel_shape)
        
    return image_est, kernel_est


# ============================================================
# DÉMO
# ============================================================

def main():
    rng = np.random.default_rng(0)

    original = img_as_float(imread("./imgs/camera.png", as_gray=True))
    
    n_small, n_main = 16, 256

    img_main = resize(original, (n_main, n_main), anti_aliasing=True)
    img_small = resize(original, (n_small, n_small), anti_aliasing=True)

    k_true = gaussian_kernel(5, 1.2)

    y_main = circular_convolve2d(img_main, k_true)
    y_main = add_gaussian_noise(y_main, sigma=0.002, rng=rng)

    y_small = circular_convolve2d(img_small, k_true)
    y_small = add_gaussian_noise(y_small, sigma=0.002, rng=rng)


    print("\n" + "=" * 60)
    print("Déconvolution aveugle avec CasADi")
    f_joint, k_joint = solve_blind_joint_casadi(
        y_small,
        kernel_shape=(5, 5),
        init_kernel=None,
    )

    print("\n" + "=" * 60)
    print("Minimisation alternée")
    f_alt, k_alt = solve_ADM(
        y_main,
        kernel_shape=(5, 5),
        wavename="db4",
        outer_iter=10,
        init_kernel=None,
    )
    
    print("\n" + "=" * 60)
    print("Richardson-Lucy blind")
    frl_blind, k_blind = blind_richardson_lucy(
        y_main,
        kernel_shape=(5, 5),
        outer_iter=30,
        init_image=None,
        init_kernel=None,
    )

    print("=" * 60)
    print("MSE image floutée (256x256):", mse(img_main, y_main))
    print("MSE image floutée (12x12):", mse(img_small, y_small))


    if f_joint is not None and k_joint is not None:
        print("\n" + "=" * 10)
        print("MSE aveugle:", mse(img_small, f_joint))
        print("Somme noyau estimé:", k_joint.sum())
        print("Valeur centrale noyau estimé:", k_joint[2, 2])

    print("\n" + "=" * 10)
    print("MSE alterné:", mse(img_main, f_alt))
    print("Somme noyau alterné:", k_alt.sum())
    print("Valeur centrale noyau alterné:", k_alt[2, 2])

    
    print("\n" + "=" * 10)
    print("MSE RL aveugle :", mse(img_main, frl_blind))
    print("Somme noyau estimé :", k_blind.sum())
    print("Valeur centrale noyau estimé :", k_blind[4, 4])

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 4, 1)
    plt.title("Original 12x12")
    plt.imshow(img_small, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("Observed 12x12")
    plt.imshow(y_small, cmap="gray")
    plt.axis("off")


    plt.subplot(2, 4, 3)
    plt.title("Convolution Aveugle CasADi")
    if f_joint is not None:
        plt.imshow(f_joint, cmap="gray")
    else:
        plt.text(0.5, 0.5, "échec", ha="center", va="center")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.title("Original 256x256")
    plt.imshow(img_main, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.title("Observed 256x256")
    plt.imshow(y_main, cmap="gray")
    plt.axis("off")

    
    plt.subplot(2, 4, 7)
    plt.title("Convolution Aveugle ADM")
    plt.imshow(f_alt, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title("Richardson-Lucy")
    plt.imshow(frl_blind, cmap="gray")
    plt.axis("off")


    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.title("Noyau vrai")
    plt.imshow(k_true, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Noyau convolution aveugle CasADi")
    if k_joint is not None:
        plt.imshow(k_joint, cmap="gray")
    else:
        plt.text(0.5, 0.5, "échec", ha="center", va="center")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Noyau convolution aveugle ADM")
    plt.imshow(k_alt, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()