import numpy as np
from . import wavelet

# ------------------------ Filtering tools -------------------------------------

def convolution(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Perform a 1D circular convolution between a signal and a filter.

    The signal is extended periodically in order to avoid boundary effects,
    and the result is cropped to preserve the original length of the signal.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input signal of length N.

    f : np.ndarray
        One-dimensional convolution filter of length L.

    Returns
    -------
    y : np.ndarray
        One-dimensional array of length N corresponding to the circular
        convolution of `x` with `f`.
    """
    pass


def conv(img: np.ndarray, wavename: str):
    """
    Apply a 1D wavelet analysis filter bank column-wise to a 2D image,
    followed by dyadic subsampling.

    This function performs the first step of a separable 2D wavelet transform.

    Parameters
    ----------
    img : np.ndarray
        2D input array of shape (NX, NY).

    wavename : str
        Name of the wavelet defining the analysis filters.

    Returns
    -------
    approx : np.ndarray
        Low-pass filtered and vertically subsampled image
        of shape (NX/2, NY).

    detail : np.ndarray
        High-pass filtered and vertically subsampled image
        of shape (NX/2, NY).
    """
    pass


def iconv_low(x: np.ndarray, wavename: str):
    """
    Low-pass synthesis filtering with upsampling.

    This function corresponds to the inverse operation of the low-pass
    analysis filter, used during wavelet reconstruction.

    Parameters
    ----------
    x : np.ndarray
        2D array of low-pass coefficients.

    wavename : str
        Name of the wavelet defining the synthesis filter.

    Returns
    -------
    y : np.ndarray
        Reconstructed signal after upsampling and low-pass filtering.
    """
    pass


def iconv_high(x: np.ndarray, wavename: str):
    """
    High-pass synthesis filtering with upsampling.

    This function reconstructs high-frequency components during
    the inverse wavelet transform.

    Parameters
    ----------
    x : np.ndarray
        2D array of high-pass coefficients.

    wavename : str
        Name of the wavelet defining the synthesis filter.

    Returns
    -------
    y : np.ndarray
        Reconstructed signal after upsampling and high-pass filtering.
    """
    pass


# ------------------- Discrete Wavelet Transform (2D) --------------------------

def dwt2D(x: np.ndarray, wavename: str, dec_level=3):
    """
    Compute the 2D Discrete Wavelet Transform (DWT) of an image.

    The transform is implemented using separable 1D wavelet filter banks
    applied successively along columns and rows.

    Parameters
    ----------
    x : np.ndarray
        2D input image of shape (NX, NY).

    wavename : str
        Name of the wavelet used for decomposition.

    dec_level : int or float, optional
        Maximum number of decomposition levels.
        Default is 3.

    Returns
    -------
    coeff : list
        Multiscale wavelet coefficient structure:
        - coeff[0] to coeff[L-1]: detail coefficients [cH, cV, cD]
          at each scale (from finest to coarsest),
        - coeff[-1]: list containing the final coarse approximation [cA].
    """
    pass


def idwt2D(coeff: list, wavename: str):
    """
    Reconstruct a 2D image from its wavelet coefficients.

    This function performs the inverse 2D discrete wavelet transform
    using separable synthesis filter banks.

    Parameters
    ----------
    coeff : list
        Wavelet coefficient structure as returned by `dwt2D`.

    wavename : str
        Name of the wavelet used for reconstruction.

    Returns
    -------
    x : np.ndarray
        Reconstructed 2D image.
    """
    pass

# -------------------- Vector / scale representations --------------------------

def vectorRepresentation(coeff: list) -> np.ndarray:
    """
    Convert a multiscale wavelet coefficient representation into
    a single one-dimensional vector.

    Parameters
    ----------
    coeff : list
        Wavelet coefficient list as returned by `dwt2D`.

    Returns
    -------
    vec : np.ndarray
        Flattened vector containing all wavelet coefficients
        concatenated scale by scale.
    """
    return np.concatenate([band.ravel() for scale in coeff for band in scale])


def scaleRepresentation(vec: np.ndarray, shape: tuple, dec_level=np.inf)-> list:
    """
    Reconstruct a multiscale wavelet coefficient structure from
    a vectorized representation.

    Parameters
    ----------
    vec : np.ndarray
        One-dimensional array of wavelet coefficients.

    shape : tuple
        Shape (NX, NY) of the original image.

    dec_level : int or float, optional
        Maximum number of decomposition levels.

    Returns
    -------
    coeff : list
        Wavelet coefficient structure compatible with `idwt2D`.
    """
    NX, NY = shape
    idx = 0
    coeff = []
    level = 0

    while level < dec_level and NX > 1 and NY > 1:
        NX //= 2
        NY //= 2
        N = NX * NY

        cH = vec[idx:idx + N].reshape(NX, NY)
        cV = vec[idx + N:idx + 2 * N].reshape(NX, NY)
        cD = vec[idx + 2 * N:idx + 3 * N].reshape(NX, NY)

        coeff.append([cH, cV, cD])
        idx += 3 * N
        level += 1

    coeff.append([vec[idx:].reshape(NX, NY)])
    return coeff


# ---------------------------- Visualization -----------------------------------

def display_transform(coeff: list) -> np.ndarray:
    """
    Create a 2D visualization of wavelet coefficients following
    the standard LL/LH/HL/HH layout.

    Parameters
    ----------
    coeff : list
        Wavelet coefficient list as returned by `dwt2D`.

    Returns
    -------
    img : np.ndarray
        Image-like array suitable for visualization.
    """
    def normalize(arr):
        if arr.size == 1:
            return arr[0, 0]
        a, b = arr.min(), arr.max()
        if b > a:
            return 255 * (arr - a) / (b - a)
        return np.zeros_like(arr)

    NX, NY = coeff[0][0].shape
    img = np.zeros((2 * NX, 2 * NY))

    LX, LY = coeff[-1][0].shape
    img[:LX, :LY] = normalize(coeff[-1][0])

    for scale in reversed(coeff[:-1]):
        cH, cV, cD = scale

        img[LX:2 * LX, :LY] = normalize(cH)
        img[:LX, LY:2 * LY] = normalize(cV)
        img[LX:2 * LX, LY:2 * LY] = normalize(cD)

        img[LX, :] = 255
        img[:, LY] = 255

        LX *= 2
        LY *= 2

    return img


# -------------------------- Compression ---------------------------------------

def dwt2D_compression(x: np.ndarray, 
    wavename: str, 
    dec_level: int
) -> np.ndarray:
    """
    Insert here your compression code using the wavelet decomposition of the
    image. Fell free to add any parameter to the function

    Parameters
    ----------
    x : np.ndarray
        Input image.

    wavename : str
        Wavelet used for the transform.


    Returns
    -------
    res : np.ndarray
        Reconstructed compressed image.
    """
    pass


# ---------------------------- Denoising ---------------------------------------

def dwt2D_denoising(x: np.ndarray, 
    wavename: str, 
    dec_level: int
) -> np.ndarray:
    """
    Insert here your denoising code using the wavelet decomposition of the
    image

    Parameters
    ----------
    x : np.ndarray
        Noisy input image.

    wavename : str
        Wavelet used for the transform.

    Returns
    -------
    res : np.ndarray
        Denoised reconstructed image.
    """
    pass

