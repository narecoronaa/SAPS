# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""    
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt


def process_fft(fs, N, senial):
    
    
    #################### Cálculo de la Transformada de Fourier ###################
    
    freq = fft.fftfreq(N, d=1/fs)   # se genera el vector de frecuencias
    senial_fft = fft.fft(senial)    # se calcula la transformada rápida de Fourier
    
    # El espectro es simétrico, nos quedamos solo con el semieje positivo
    f = freq[np.where(freq >= 0)]      
    senial_fft = senial_fft[np.where(freq >= 0)]
    
    # Se calcula la magnitud del espectro
    senial_fft_mod = np.abs(senial_fft) / N     # Respetando la relación de Parseval
    # Al haberse descartado la mitad del espectro, para conservar la energía 
    # original de la señal, se debe multiplicar la mitad restante por dos (excepto
    # en 0 y fm/2)
    senial_fft_mod[1:len(senial_fft_mod-1)] = 2 * senial_fft_mod[1:len(senial_fft_mod-1)]
    
    # Devuelve el modulo de la fft de la señal junto con su frecuencia
    return f,senial_fft_mod
    
    # plt.plot(f,senial_fft_mod)
    # plt.xlabel("Frecuencia[Hz]")
    # plt.ylabel("Magnitud")
    # plt.title('Espectro de Magnitud')
    # plt.grid()
    # plt.show()