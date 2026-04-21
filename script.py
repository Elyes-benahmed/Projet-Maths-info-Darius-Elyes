import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_float
from src.dwt import *


if __name__ == "__main__":

    # Configuration
    original = img_as_float(imread('./imgs/camera.png'))
    WAVENAME = 'db8'
    
    # ==== Compute and display the wavelet decomposition of the image ====
    coefs = dwt2D(original, WAVENAME, dec_level=2)
    coefs_display = display_transform(coefs)
    plt.figure()
    plt.title("Wavelet Decomposition")
    plt.imshow(coefs_display, cmap='gray')
    plt.axis('off')
    plt.show()

    # ==== Check vector representation ====
    vec = vectorRepresentation(coefs)
    coefs_test = scaleRepresentation(vec, shape =(512, 512), dec_level=2) 
    
        
    # ==== Reconstruction ====
    reconstruction = idwt2D(coefs, WAVENAME)
    mse_reconstruction = np.mean((original - reconstruction) ** 2)
    print(f"Compression MSE: {mse_reconstruction:.15f}")

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    plt.imshow(reconstruction, cmap='gray')
    plt.axis('off')
    plt.show()
        
    # ==== Compression ====
    compression_rate = 0.1
    k = int(compression_rate * original.size)
    compressed = dwt2D_compression(original, WAVENAME, k, dec_level=2)
    mse_compressed = np.mean((original - compressed) ** 2)
    print(f"Reconstruction MSE: {mse_compressed:.6f}")
        
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Compressed")
    plt.imshow(compressed, cmap='gray')
    plt.axis('off')
    plt.show()
      
    # ==== Denoising ====  
    noisy = img_as_float(imread('./imgs/noisy.png'))
    mse_noisy = np.mean((noisy - original) ** 2)
    print(f"Noisy MSE: {mse_noisy:.6f}")
        
    denoised = dwt2D_denoising(noisy, 'db4', dec_level=4)
    mse_denoised = np.mean((original - denoised) ** 2)
    print(f"Denoised MSE: {mse_denoised:.6f}")
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1) 
    plt.title("Noisy")
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Denoised")
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')
    plt.show()

