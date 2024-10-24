from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure, img_as_ubyte, filters
from skimage.restoration import denoise_bilateral
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.signal import fftconvolve

def ajustar_contraste_e_filtro(image_file):
    """
    Carrega uma imagem astronômica, aplica filtros, ajusta o contraste e a nitidez.
    """
    # Carrega a imagem FITS
    hdul = fits.open(image_file)
    image_data = hdul[0].data
    hdul.close()

    # Garante que os valores na imagem original sejam não negativos
    image_data = np.clip(image_data, 0, np.inf) 

    # Ajusta a intensidade da imagem para o intervalo [0, 255]
    image_data = exposure.rescale_intensity(image_data, out_range=(0, 255))
    image_data = np.clip(image_data, 0, 255) # Clip após reescalar

    # Normaliza a imagem para o intervalo [0, 1] ANTES de converter para uint8
    image_data = image_data / 255.0 

    # Converte a imagem para uint8
    uint8_image = img_as_ubyte(image_data)

    # Aplica filtro mediana para suavizar a imagem
    median_filtered = median_filter(uint8_image, size=3)  # Ajusta o tamanho do kernel
    
    # Aplica filtro gaussiano com sigma maior para suavizar a imagem
    gaussian_filtered = gaussian_filter(median_filtered, sigma=3)  # Aumenta o sigma

    # Aplica CLAHE
    clahe = exposure.equalize_adapthist(median_filtered, clip_limit=0.03)  # Ajusta o clip_limit
    clahe = np.clip(clahe, 0, 255) # Clip após CLAHE

    # Aplica Unsharp Masking
    unsharp_image = filters.unsharp_mask(clahe, radius=0.8, amount=2.5)  # Ajusta radius e amount
    unsharp_image = np.clip(unsharp_image, 0, 255) # Clip após Unsharp Masking
    
    # Aumenta o brilho de forma gradual
    adjusted_image = clahe + 20  # Aumenta a constante para clarear mais o fundo
    adjusted_image = exposure.rescale_intensity(adjusted_image, out_range=(0, 255))  # Normaliza

    # Clareia a galáxia
    adjusted_image = unsharp_image * 2.5  # Ajusta a constante para clarear a galáxia
    
    # Escurece o fundo
    adjusted_image = adjusted_image - 5  # Ajusta a constante para escurecer o fundo
    adjusted_image = exposure.rescale_intensity(adjusted_image, out_range=(0, 255))  # Normaliza

    # Aplica ajuste de curvas (exemplo simplificado) com gamma ajustado
    adjusted_image = exposure.adjust_gamma(adjusted_image, gamma=1.3) # Gamma ajustado



    
    
    # Exibe a imagem
    plt.figure(figsize=(8, 8))
    plt.imshow(adjusted_image, cmap='gray')
    plt.title('Imagem com Contraste Ajustado e Filtro de Mediana')
    plt.axis('off')
    plt.show()

# Substitua pelo caminho da sua imagem
image_file = 'ui.3007920713.1_0.fits'

# Chama a função para processar a imagem
ajustar_contraste_e_filtro(image_file)
