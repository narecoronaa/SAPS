# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Analógico:
    En el siguiente script se ejemplifica el proceso de análisis de señales 
    para la definición de requisitos para un filtro antialiasing.
    También se ejemplifica la importación de los resultados del diseño de filtros
    utilizando Analog Filter Wizard y de simulaciones realizadas mediante
    LTSpice.

Autor: Albano Peñalva
Fecha: Febrero 2025

"""

# Librerías
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import process_data
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
from funciones_fft import fft_mag

plt.close('all') # cerrar gráficas anteriores

#%% Lectura del dataset

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 2 segundos

folder = 'dataset_voley' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)
print("\r")

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos

#%% Análisis frecuencial de las señales

SENS = 300      # Sensibilidad del ADXL335 [mV/g]
OFF  = 1650     # Offset del ADXL335 [mV]

# A modo de ejemplo se calcula la FFT sobre una sola señal (la primera en el eje x)
# (este análisis se debería repetir para todas las señales, buscando el peor caso).
f, h = fft_mag(x[0, 0:N]*SENS+OFF, FS)

# Se grafica la FFT
fig2, ax2 = plt.subplots(1, 1, figsize=(18, 12))
ax2.set_title('FFT', fontsize=18)
ax2.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2.set_ylabel('Magnitud [mV]', fontsize=15)
ax2.set_xlim(0, 100)
# ax2.set_ylim(0, 20)
ax2.grid(True, which="both")
# ax2.plot(f, h, label='FFT')

# Se propone una nueva frecuencia de muestreo para el sistema
FS2 = 60                                    # Nueva frecuencia de muestreo: 60Hz
fs2_2 = f[np.where(f>=(FS2/2))][0]          # Valor más cercano a FS2/2

# Para determinar los requerimientos del antialiasing, primero analizamos el 
# contenido espectral de las señales por encima de FS2/2 en dos puntos (peores casos):
    
# Donde se encuentre el máximo a partir de FS2/2
h_max = np.max( h [np.where(f >= fs2_2) ] )
f_max = f[ np.argmax(h[np.where(f >= fs2_2)])] + fs2_2
# Exactamente en FS2/2
h_fs_2 = np.max( h [np.where(f == fs2_2) ] )
f_fs_2 = f[ np.argmax( h[ np.where(f == fs2_2) ] ) ] + fs2_2

# print(f"Interferencia de {h_max:.2f}mV en {f_max}Hz")
# print(f"Interferencia de {h_fs_2:.2f}mV en {f_fs_2}Hz")
# print("\r")

ax2.axvline(x=FS2/2, color="black", linestyle="--")
ax2.plot(f_max, h_max, marker='X', markersize=12, label='Amplitud en fs/2')
ax2.plot(f_fs_2, h_fs_2, marker='X', markersize=12, label='Máximo a partir de fs/2')
ax2.legend(loc='upper right');

#%% Determinar requerimienos del filtro antialiasing

# Parámetros ADC:
V_REF = 3300        # Tensión de referencia en mV
N_BITS = 12         # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en mV

# Atenuaciones necesarias 
at_max = 20*np.log10(h_max/RES) 
at_fs2_2 = 20*np.log10(h_fs_2/RES)

print("Banda de paso hasta 25Hz")   # Banda de paso determinada en guía 1
print(f"Atenuación mayor a {at_max:.2f}dB en {f_max}Hz")
print(f"Atenuación mayor a {at_fs2_2:.2f}dB en {f_fs_2}Hz")
print("\r")

#%% Importar resultados de Diseño de Analog Filter Wizard
f, mag = import_AnalogFilterWizard('DesignFilesEjemplo/Data Files/Magnitude(dB).csv')


#%% Importar resultados de simulación en LTSpice
f_sim, mag_sim, _ = import_AC_LTSpice('DesignFilesEjemplo/SPICE Files/LTspice/ACAnalysis.txt')

# Análisis de la atenuación del filtro simulado en las frecuencias de interés
F_AT1 = FS2/2
F_AT2 = f_max
# se calcula la atenuación en el punto mas cercano a la frecuencia de interés
at1 = mag_sim[np.argmin(np.abs(f_sim-F_AT1))] 
print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(F_AT1, at1))
at2 = mag_sim[np.argmin(np.abs(f_sim-F_AT2))] 
print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(F_AT2, at2))
print("\r")

#%% Comparación de las respuestas en frecuencia del filtro diseñado y el simulado 

# Se crea una gráfica para comparar los filtros 
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))

ax3.set_title('Filtro orden 4', fontsize=18)
ax3.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3.set_xscale('log')
ax3.grid(True, which="both")
ax3.plot(f,  mag, label='Diseñado')
ax3.plot(f_sim,  mag_sim, label='Simulado')
ax3.plot(f_fs_2, -at_fs2_2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(f_max, -at_max, marker='X', markersize=12, label='Requisito en máximo a partir de fs/2')
ax3.legend(loc="lower left", fontsize=15)

#%% Comparación con la respuestas en frecuencia del filtro implementado

f_impl = [1, 2, 5, 10, 11, 12, 15, 20, 21, 22, 25, 50, 100, 200, 500]
mag_impl = [0.0, 0.5, 1.5, 2.0, 1.5, 1.0, 0.0, 1.5, 2.0, 1.5, 0.0, -30, -60, -80, -90]
ax3.plot(f_impl,  mag_impl, label='Implementado')

plt.show()