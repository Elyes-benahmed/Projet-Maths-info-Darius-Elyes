import numpy as np
from . import wavelet

# ------------------------ Filtering tools -------------------------------------


def convolution(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Perform a 1D circular convolution between a signal and a filter.
    ...
    """
    x = np.asarray(x).ravel()
    f = np.asarray(f).ravel()

    N = len(x)
    L = len(f)
    c = L // 2  # Centre du filtre (décalage pour centrer la réponse)

    y = np.zeros(N, dtype=np.result_type(x, f))

    # Convolution discrète 
    # np.roll assure l'extension circulaire (pas de zero-padding aux bords)
    for k in range(L):
        y += f[k] * np.roll(x, c - k)

    return y


def conv(img: np.ndarray, wavename: str):
    """
    Apply a 1D wavelet analysis filter bank column-wise to a 2D image,
    followed by dyadic subsampling.
    ...
    """
    NX, NY = img.shape
    w = wavelet.wavelet(wavename)  # Chargement des filtres d'analyse du wavelet

    # Tableaux de sortie sous-échantillonnés d'un facteur 2 en vertical
    approx = np.zeros((NX // 2, NY))
    detail = np.zeros((NX // 2, NY))

    for j in range(NY):
        # Filtrage passe-bas puis décimation par 2 (échantillons pairs)
        approx[:, j] = convolution(img[:, j], np.array(w.dec_low))[1::2]
        # Filtrage passe-haut puis décimation par 2
        detail[:, j] = convolution(img[:, j], np.array(w.dec_high))[1::2]

    return approx, detail


def iconv_low(x: np.ndarray, wavename: str):
    """
    Low-pass synthesis filtering with upsampling.
    ...
    """
    x = np.asarray(x)
    NX, NY = x.shape

    w = wavelet.wavelet(wavename)
    h = np.array(w.rec_low)  # Filtre de synthèse passe-bas

    y = np.zeros((2 * NX, NY), dtype=np.result_type(x, h))

    for j in range(NY):
        # Sur-échantillonnage par 2 : insertion de zéros entre chaque échantillon
        up = np.zeros(2 * NX, dtype=np.result_type(x, h))
        up[::2] = x[:, j]  # Les échantillons originaux vont aux positions paires
        y[:, j] = convolution(up, h)  # Filtrage passe-bas pour interpoler les zéros

    return y


def iconv_high(x: np.ndarray, wavename: str):
    """
    High-pass synthesis filtering with upsampling.
    ...
    """
    x = np.asarray(x)
    NX, NY = x.shape

    w = wavelet.wavelet(wavename)
    g = np.array(w.rec_high)  # Filtre de synthèse passe-haut

    y = np.zeros((2 * NX, NY), dtype=np.result_type(x, g))

    for j in range(NY):
        # Sur-échantillonnage par 2 (même principe que iconv_low)
        up = np.zeros(2 * NX, dtype=np.result_type(x, g))
        up[::2] = x[:, j]
        y[:, j] = convolution(up, g)  # Filtrage passe-haut pour reconstruire les détails

    return y


# ------------------- Discrete Wavelet Transform (2D) --------------------------

def dwt2D(x: np.ndarray, wavename: str, dec_level=3):
    """
    Compute the 2D Discrete Wavelet Transform (DWT) of an image.
    ...
    """
    x = np.asarray(x)
    coeff = []
    current = x.copy()
    level = 0

    # Décomposition itérative : on raffine jusqu'au niveau demandé
    # ou tant que l'image est suffisamment grande pour être subdivisée
    while level < dec_level and current.shape[0] > 1 and current.shape[1] > 1:

        # --- Étape 1 : filtrage colonne par colonne ---
        # low_col  : composante basse fréquence verticale
        # high_col : composante haute fréquence verticale
        low_col, high_col = conv(current, wavename)

        # --- Étape 2 : filtrage ligne par ligne (via transposition) ---
        # On transpose pour réutiliser conv() qui opère sur les colonnes
        # cA : approximation (LL), cV : détails verticaux (LH)
        cA, cV = conv(low_col.T, wavename)
        # cH : détails horizontaux (HL), cD : détails diagonaux (HH)
        cH, cD = conv(high_col.T, wavename)

        # Retour à l'orientation image originale
        cA = cA.T
        cV = cV.T
        cH = cH.T
        cD = cD.T

        # On stocke les détails [cH, cV, cD] du niveau courant (du plus fin au plus grossier)
        coeff.append([cH, cV, cD])

        # L'approximation devient l'entrée du niveau suivant
        current = cA
        level += 1

    # Dernier élément : approximation grossière finale (sous-bande LL)
    coeff.append([current])
    return coeff


def idwt2D(coeff: list, wavename: str):
    """
    Reconstruct a 2D image from its wavelet coefficients.
    ...
    """
    # On part de l'approximation la plus grossière (dernier élément)
    current = coeff[-1][0]

    # Remontée de l'échelle la plus grossière vers la plus fine
    for scale in reversed(coeff[:-1]):
        cH, cV, cD = scale

        # --- Reconstruction le long des lignes (via transposition) ---
        # Branche basse : approximation LL + détails verticaux LH
        low_rows  = iconv_low(current.T, wavename) + iconv_high(cV.T, wavename)
        # Branche haute : détails horizontaux HL + détails diagonaux HH
        high_rows = iconv_low(cH.T, wavename)      + iconv_high(cD.T, wavename)

        low_rows  = low_rows.T
        high_rows = high_rows.T

        # --- Reconstruction le long des colonnes ---
        current = iconv_low(low_rows, wavename) + iconv_high(high_rows, wavename)

    return current


# -------------------- Vector / scale representations --------------------------

def vectorRepresentation(coeff: list) -> np.ndarray:
    """
    Convert a multiscale wavelet coefficient representation into
    a single one-dimensional vector.
    ...
    """
    # Concatène à plat toutes les sous-bandes (cH, cV, cD, puis cA final)
    # dans l'ordre de coeff, du niveau le plus fin au plus grossier
    return np.concatenate([band.ravel() for scale in coeff for band in scale])


def scaleRepresentation(vec: np.ndarray, shape: tuple, dec_level=np.inf) -> list:
    """
    Reconstruct a multiscale wavelet coefficient structure from
    a vectorized representation.
    ...
    """
    NX, NY = shape
    idx = 0       
    coeff = []
    level = 0

    while level < dec_level and NX > 1 and NY > 1:
        # La taille des sous-bandes est divisée par 2 à chaque niveau
        NX //= 2
        NY //= 2
        N = NX * NY  # Nombre de coefficients par sous-bande à ce niveau

        # Extraction des trois sous-bandes de détails depuis le vecteur
        cH = vec[idx:idx + N].reshape(NX, NY)
        cV = vec[idx + N:idx + 2 * N].reshape(NX, NY)
        cD = vec[idx + 2 * N:idx + 3 * N].reshape(NX, NY)

        coeff.append([cH, cV, cD])
        idx += 3 * N  # On avance de 3 sous-bandes
        level += 1

    # Le reste du vecteur correspond à l'approximation finale
    coeff.append([vec[idx:].reshape(NX, NY)])
    return coeff


# ---------------------------- Visualization -----------------------------------

def display_transform(coeff: list) -> np.ndarray:
    """
    Create a 2D visualization of wavelet coefficients following
    the standard LL/LH/HL/HH layout.
    ...
    """
    def normalize(arr):
        """Normalise un tableau dans [0, 255] pour l'affichage."""
        if arr.size == 1:
            return arr[0, 0]
        a, b = arr.min(), arr.max()
        if b > a:
            return 255 * (arr - a) / (b - a)
        return np.zeros_like(arr)  # Tableau constant → tout à 0

    # Taille du niveau le plus fin (premier niveau de détails)
    NX, NY = coeff[0][0].shape
    img = np.zeros((2 * NX, 2 * NY))  # Image de sortie au double de la résolution

    # Placement de l'approximation grossière en haut à gauche
    LX, LY = coeff[-1][0].shape
    img[:LX, :LY] = normalize(coeff[-1][0])

    # Placement des sous-bandes de détail de l'échelle grossière vers la fine
    for scale in reversed(coeff[:-1]):
        cH, cV, cD = scale

        img[LX:2 * LX, :LY]     = normalize(cH)  # Détails horizontaux (bas-gauche)
        img[:LX, LY:2 * LY]     = normalize(cV)  # Détails verticaux   (haut-droite)
        img[LX:2 * LX, LY:2 * LY] = normalize(cD)  # Détails diagonaux   (bas-droite)

        # Lignes de séparation blanches entre les quadrants
        img[LX, :] = 255
        img[:, LY] = 255

        # On double la fenêtre pour passer au niveau suivant (plus fin)
        LX *= 2
        LY *= 2

    return img


# -------------------------- Compression ---------------------------------------

def dwt2D_compression(
    x: np.ndarray,
    wavename: str,
    k: int,
    dec_level: int
) -> np.ndarray:
    """
    Compress an image by keeping only the k largest wavelet coefficients
    in absolute value.
    ...
    """
    coeff = dwt2D(x, wavename, dec_level=dec_level)
    vec = vectorRepresentation(coeff).copy()

    # Sécurisation : k doit être dans [0, nombre total de coefficients]
    k = max(0, min(k, vec.size))

    if k == 0:
        # Compression totale : on annule tous les coefficients
        vec[:] = 0
    elif k < vec.size:
        # Calcul du seuil = valeur absolue du k-ième plus grand coefficient
        threshold = np.partition(np.abs(vec), -k)[-k]
        # Mise à zéro de tous les coefficients en dessous du seuil
        vec[np.abs(vec) < threshold] = 0

        # Gestion des égalités : np.partition peut retenir plus de k coefficients
        # si plusieurs ont exactement la valeur seuil → on force exactement k
        nz = np.flatnonzero(vec)
        if nz.size > k:
            order = np.argsort(np.abs(vec[nz]))[::-1]  # Tri décroissant par valeur absolue
            keep = nz[order[:k]]                        # On garde les k plus grands
            new_vec = np.zeros_like(vec)
            new_vec[keep] = vec[keep]
            vec = new_vec

    # Reconstruction depuis le vecteur sparse
    coeff_compressed = scaleRepresentation(vec, x.shape, dec_level=dec_level)
    res = idwt2D(coeff_compressed, wavename)

    # Recadrage dans [0, 1] pour rester dans la plage d'une image normalisée
    return np.clip(res, 0.0, 1.0)


def dwt2D_denoising(
    x: np.ndarray,
    wavename: str,
    dec_level: int
) -> np.ndarray:
    """
    Denoise an image using wavelet HARD-thresholding.
    """
    coeff = dwt2D(x, wavename, dec_level=dec_level)

    # --- Estimation du bruit ---
    _, _, cD1 = coeff[0]
    sigma = np.median(np.abs(cD1)) / 0.6745

    # --- Seuil ---
    N = x.size
    T = sigma * np.sqrt(2 * np.log(N))

    coeff_denoised = []

    # --- Seuillage dur ---
    for cH, cV, cD in coeff[:-1]:
        cH_d = hard_threshold(cH, T)
        cV_d = hard_threshold(cV, T)
        cD_d = hard_threshold(cD, T)

        coeff_denoised.append([cH_d, cV_d, cD_d])

    # --- Approximation conservée ---
    coeff_denoised.append([coeff[-1][0]])

    res = idwt2D(coeff_denoised, wavename)
    return np.clip(res, 0.0, 1.0)


def hard_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard-threshold:
    met à 0 les coefficients de faible amplitude.
    """
    out = arr.copy()
    out[np.abs(out) < threshold] = 0.0
    return out
