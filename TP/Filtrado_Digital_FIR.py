# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Digital:
    En el siguiente script se ejemplifica el proceso de carga de filtros 
    digitales creados con la herramienta pyFDA y el uso de los mismos para el 
    filtrado de señales.

Autor: Albano Peñalva
Fecha: Febrero 2025

"""

# Librerías
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import funciones_fft
from time import time
import sympy as sy
import process_data
import filter_parameters
import process_code


plt.close('all') # cerrar gráficas anteriores

#%% Lectura del dataset

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 3 segundos


folder = 'dataset_voley' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)
print("\r\n")

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos

#%% Cálculo y Graficación de la Transformada de Fourier

FIR1 ='Filtrado FIR - Equiripple'
FIR2 ='Filtrado FIR - Windowed'
carpeta = r'C:\Users\narec\Documents\SAPS\TP\Filtros_Digitales_Graficas'

# Parámetros para el remuestreo de las señales
FS_resample = 60                        # Frecuencia de muestreo para la cual estan diseñados los filtros
N_resample = int(T*FS_resample)         # Longitud de las señales remuestreadas

# A modo de ejemplo se levanta la primer captura del eje Z del Dataset 
senial = signal.resample(z[0, 0:N], N_resample)
t_resampled = np.linspace(0, N_resample / FS_resample, N_resample)

# Se crea una gráfica 
fig1, ax1 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig1.suptitle("Señal de aceleración", fontsize=18)

# Se grafica la señal
ax1[0].plot(t_resampled, senial, label='Señal Contaminada')
ax1[0].set_ylabel('Tensión [V]', fontsize=15)
ax1[0].grid()
ax1[0].legend(loc="upper right", fontsize=15)
ax1[0].set_title(FIR1, fontsize=15)
ax1[1].plot(t_resampled, senial, label='Señal Contaminada')
ax1[1].set_ylabel('Tensión [V]', fontsize=15)
ax1[1].grid()
ax1[1].legend(loc="upper right", fontsize=15)
ax1[1].set_xlabel('Tiempo [s]', fontsize=15)
ax1[1].set_xlim([0, ts*N])
ax1[1].set_title(FIR2, fontsize=15)

# Se calcula el espectro de la señal contaminada
f, senial_fft_mod = funciones_fft.fft_mag(senial, FS_resample)

# Se crea una gráfica 
fig2, ax2 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig2.suptitle("Señal de aceleración", fontsize=18)

# Se grafica la magnitud del espectro (normalizado)
ax2[0].plot(f, senial_fft_mod, label='Señal Original')
ax2[0].set_ylabel('Magnitud', fontsize=15)
ax2[0].grid()
ax2[0].legend(loc="upper right", fontsize=15)
ax2[0].set_title(FIR1, fontsize=15)
ax2[1].plot(f, senial_fft_mod, label='Señal Original')
ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[1].grid()
ax2[1].legend(loc="upper right", fontsize=15)
ax2[1].set_title(FIR2, fontsize=15)
ax2[1].set_ylabel('Magnitud', fontsize=15)
ax2[1].set_xlim([0, FS_resample/2])

#%% Carga de los Filtros 

# Se cargan los archivo generado mediante pyFDA
filtro_fir1 = np.load('FIR_Equiriripple-O109.npz', allow_pickle=True)
filtro_fir2 = np.load('FIR_WindowedFIR-O74.npz', allow_pickle=True) 

# Se muestran parámetros de diseño
print("FILTRO FIR 1: " + FIR1)
filter_parameters.filter_parameters('FIR_Equiriripple-O109.npz')
print("\r\n")
print("Filtro FIR 2: " + FIR2)
filter_parameters.filter_parameters('FIR_WindowedFIR-O74.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_fir1, Den_fir1 = filtro_fir1['ba']     
Num_fir2, Den_fir2 = filtro_fir2['ba'] 

# Se expresan las funciones de transferencias (H(z))
z_sym = sy.Symbol('z') # Se crea una variable simbólica z
Hz = sy.Symbol('H(z)')

# Función de transferencia FIR1
Numz_fir1 = 0
Denz_fir1 = 0
for i in range(len(Num_fir1)): # Se arma el polinomio del numerador
    Numz_fir1 += Num_fir1[i] * np.power(z_sym, -i)
for i in range(len(Den_fir2)): # Se arma el polinomio del denominador
    Denz_fir1 += Den_fir1[i] * np.power(z_sym, -i)
# print("La función de transferencia del Filtro IIR es:")
# print(sy.pretty(sy.Eq(Hz, Numz_fir1.evalf(3) / Denz_fir1.evalf(3))))
# print("\r\n")

# Función de transferencia FIR2
Numz_fir2 = 0
Denz_fir2 = 0
for i in range(len(Num_fir2)): # Se arma el polinomio del numerador
    Numz_fir2 += Num_fir2[i] * np.power(z_sym, -i)
for i in range(len(Den_fir2)): # Se arma el polinomio del denominador
    Denz_fir2 += Den_fir2[i] * np.power(z_sym, -i)
# print("La función de transferencia del Filtro IIR es:")
# print(sy.pretty(sy.Eq(Hz, Numz_fir2.evalf(3) / Denz_fir2.evalf(3)))) 
# print("\r\n")

#%% Análisis de los Filtros 
f_log = np.logspace(-1, 2, int(10e3))  # vector de frecuencia
# Se calcula la respuesta en frecuencia de los filtros
f_fir1, h_fir1 = signal.freqz(Num_fir1, Den_fir1, worN=f_log, fs=FS_resample)
f_fir2, h_fir2 = signal.freqz(Num_fir2, Den_fir2, worN=f_log, fs=FS_resample)

# Se crea una gráfica 
fig3, ax3 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig3.suptitle("Respuesta en Frecuencia de los filtros", fontsize=18)

# Se grafican las respuestas de los filtros
ax3[0].plot(f_fir1, abs(h_fir1), label='Filtro FIR', color='orange')
ax3[0].legend(loc="upper right", fontsize=15)
ax3[0].set_title(FIR1, fontsize=15)
ax3[0].set_xlim([0, FS_resample/2])
ax3[0].grid()

ax3[1].plot(f_fir2, abs(h_fir2), label='Filtro FIR2', color='green')
ax3[1].legend(loc="upper right", fontsize=15)
ax3[1].set_title(FIR2, fontsize=15)
ax3[1].set_xlim([0, FS_resample/2])
ax3[1].grid()

#RESPUESTA IMPLEMENTADA
# Datos de frecuencia y salida (en G)
f_impl = [0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 5, 10, 11, 12, 13, 14, 15, 16, 17, 17.5, 18, 18.5]
salida_impl = [1.4, 1.5, 1.6, 2, 2.2, 2.5, 2.7, 3.2, 3.4, 3.4, 3.4, 3.4, 3, 3, 2.6, 2.5, 3.4, 2.7, 2.5, 1.6, 1, 0.5]

# Ganancia 
ganancia_impl = [y/3.4 for y in salida_impl]

# Graficar la respuesta implementada sobre el Chebyshev
ax3[1].plot(f_impl, ganancia_impl, 'o-', color='blue', label='Implementado')

# fig3.savefig(rf'{carpeta}\respuesta_en_frec_fir.png', dpi=200)

# Se evalúa la atenuación en las frecuncias de interés 
_, h1_fir1 = signal.freqz(Num_fir1, Den_fir1, worN=[0.01, 20], fs=FS_resample)
_, h1_fir2 = signal.freqz(Num_fir2, Den_fir2, worN=[0.01, 20], fs=FS_resample)

# print(f"La atenuación del {FIR1} en 0.01Hz es de {20*np.log10(abs(h1_fir1[0])):.2f}dB")
# print(f"La atenuación del {FIR1} en 20Hz es de {20*np.log10(abs(h1_fir1[1])):.2f}dB")
# print(f"La atenuación del {FIR2} en 0.01Hz es de {20*np.log10(abs(h1_fir2[0])):.2f}dB")
# print(f"La atenuación del {FIR2} en 20Hz es de {20*np.log10(abs(h1_fir2[1])):.2f}dB")
# print("\r\n")

# Se extraen polos y ceros de los filtros
zeros_fir1, polos_fir1, k_fir1 =   filtro_fir1['zpk']  
zeros_fir2, polos_fir2, k_fir2 =   filtro_fir2['zpk']

# Se grafican las distribuciones de ceros y polos
fig4, ax4 = plt.subplots(1, 2, figsize=(15, 7))
fig4.suptitle("Distribución de Ceros y Polos en el plano Z", fontsize=18)

ax4[0].set_title(FIR1, fontsize=15)
ax4[0].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax4[0].plot(polos_fir1.real, polos_fir1.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax4[0].plot(zeros_fir1.real, zeros_fir1.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_fir1)), np.max(abs(zeros_fir1))])
ax4[0].set_xlim(-lim, lim)
ax4[0].set_ylim(-lim, lim)
ax4[0].set_ylabel('Imag(z)', fontsize=15)
ax4[0].set_xlabel('Real(z)', fontsize=15)
ax4[0].grid()
ax4[0].legend(loc="upper right", fontsize=12)

ax4[1].set_title(FIR2, fontsize=15)
ax4[1].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax4[1].plot(polos_fir2.real, polos_fir2.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax4[1].plot(zeros_fir2.real, zeros_fir2.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_fir2)), np.max(abs(zeros_fir2))])
ax4[1].set_xlim(-lim, lim)
ax4[1].set_ylim(-lim, lim)
ax4[1].set_ylabel('Imag(z)', fontsize=15)
ax4[1].set_xlabel('Real(z)', fontsize=15)
ax4[1].grid()
ax4[1].legend(loc="upper right", fontsize=12)

# fig4.savefig(f'{carpeta}\ceros_polos_fir.png', dpi=200)
#%% Filtrado de la Señal 

# Se aplica el filtrado sobre la señal
senial_fir1 = signal.lfilter(Num_fir1, Den_fir1, senial)
senial_fir2 = signal.lfilter(Num_fir2, Den_fir2, senial)

# Se grafican las señales filtradas
ax1[0].plot(t_resampled, senial_fir1, label='Señal Filtrada (FIR1)', color='red')
ax1[0].legend(loc="upper right", fontsize=15)
ax1[1].plot(t_resampled, senial_fir2, label='Señal Filtrada (FIR2)', color='purple')
ax1[1].legend(loc="upper right", fontsize=15)

# Se calculan y grafican sus espectros (normalizados)
f1_fir1, senial_fir1_fft_mod = funciones_fft.fft_mag(senial_fir1, FS_resample)
f1_fir2, senial_fir2_fft_mod = funciones_fft.fft_mag(senial_fir2, FS_resample)
ax2[0].plot(f1_fir1, senial_fir1_fft_mod, 
            label='Senial Filtrada FIR1', color='red')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f1_fir2, senial_fir2_fft_mod, 
            label='Senial Filtrada FIR2', color='purple')
ax2[1].legend(loc="upper right", fontsize=15)

# fig1.savefig(rf'{carpeta}\aceleracion_contaminada_fir.png', dpi=200)
# fig2.savefig(rf'{carpeta}\aceleracion_original_fir.png', dpi=200)
plt.show()


#%% Graficación señales sin filtrar

# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig5, ax5 = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig5.subplots_adjust(hspace=0.5)
fig5.suptitle("Señales sin filtrar", fontsize=15)

# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se grafica la señal en los tres ejes
            ax5[gesture_name][0].plot(t_resampled, signal.resample(x[capture, 0:N], N_resample), label="Trial {}".format(trial_num))
            ax5[gesture_name][1].plot(t_resampled, signal.resample(y[capture, 0:N], N_resample), label="Trial {}".format(trial_num))
            ax5[gesture_name][2].plot(t_resampled, signal.resample(z[capture, 0:N], N_resample), label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

# Se le da formato a los ejes de cada gráfica
    ax5[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    ax5[gesture_name][0].grid()
    ax5[gesture_name][0].legend(fontsize=6, loc='upper right');
    ax5[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    ax5[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    ax5[gesture_name][0].set_ylim(-6, 6)
    
    ax5[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    ax5[gesture_name][1].grid()
    ax5[gesture_name][1].legend(fontsize=6, loc='upper right');
    ax5[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    ax5[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    ax5[gesture_name][1].set_ylim(-6, 6)
    
    ax5[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    ax5[gesture_name][2].grid()
    ax5[gesture_name][2].legend(fontsize=6, loc='upper right');
    ax5[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    ax5[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)
    ax5[gesture_name][2].set_ylim(-6, 6)
    
    # fig5.savefig(f'{carpeta}\señales_sinFiltrar', dpi=200)

plt.show()

#%% Graficación señales filtradas (FIR1)

# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig6, ax6 = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig6.subplots_adjust(hspace=0.5)
fig6.suptitle("Señales filtradas con filtro FIR - EQUIRIPLE", fontsize=15)

# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se filtran las señales
            x_filt = signal.lfilter(Num_fir2, Den_fir2, signal.resample(x[capture, 0:N], N_resample))
            y_filt = signal.lfilter(Num_fir2, Den_fir2, signal.resample(y[capture, 0:N], N_resample))
            z_filt = signal.lfilter(Num_fir2, Den_fir2, signal.resample(z[capture, 0:N], N_resample))
            # Se grafica la señal en los tres ejes
            ax6[gesture_name][0].plot(t_resampled, x_filt, label="Trial {}".format(trial_num))
            ax6[gesture_name][1].plot(t_resampled, y_filt, label="Trial {}".format(trial_num))
            ax6[gesture_name][2].plot(t_resampled, z_filt, label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

# Se le da formato a los ejes de cada gráfica
    ax6[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    ax6[gesture_name][0].grid()
    ax6[gesture_name][0].legend(fontsize=6, loc='upper right');
    ax6[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    ax6[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    ax6[gesture_name][0].set_ylim(-6, 6)
    
    ax6[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    ax6[gesture_name][1].grid()
    ax6[gesture_name][1].legend(fontsize=6, loc='upper right');
    ax6[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    ax6[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    ax6[gesture_name][1].set_ylim(-6, 6)
    
    ax6[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    ax6[gesture_name][2].grid()
    ax6[gesture_name][2].legend(fontsize=6, loc='upper right');
    ax6[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    ax6[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)
    ax6[gesture_name][2].set_ylim(-6, 6)
    
    # fig6.savefig(f'{carpeta}\señales_filt_fir_equiriple.png', dpi=200)

plt.show()

#%% Graficación señales filtradas (FIR2)

# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig7, ax7 = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig7.subplots_adjust(hspace=0.5)
fig7.suptitle("Señales filtradas con filtro FIR - WINDOWED", fontsize=15)

# Para compensar el efecto de retardo de los filtros FIR agregaremos ceros al final de cada señal
N_fir = len(Num_fir1)    # Cantidad de coeficientes del filtro FIR
N_fir_2 = int(N_fir/2)  # Cantidad de ceros a agregar al final de la señal

# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se concatenan ceros al final de las señales
            x_long = np.concatenate([signal.resample(x[capture, 0:N], N_resample), np.zeros(N_fir_2)])
            y_long = np.concatenate([signal.resample(y[capture, 0:N], N_resample), np.zeros(N_fir_2)])
            z_long = np.concatenate([signal.resample(z[capture, 0:N], N_resample), np.zeros(N_fir_2)])
            # Se filtran las señales
            x_filt = signal.lfilter(Num_fir1, Den_fir1, x_long)
            y_filt = signal.lfilter(Num_fir1, Den_fir1, y_long)
            z_filt = signal.lfilter(Num_fir1, Den_fir1, z_long)
            # Se grafica la señal en los tres ejes
            ax7[gesture_name][0].plot(t_resampled, x_filt[N_fir_2:N+N_fir_2], label="Trial {}".format(trial_num))
            ax7[gesture_name][1].plot(t_resampled, y_filt[N_fir_2:N+N_fir_2], label="Trial {}".format(trial_num))
            ax7[gesture_name][2].plot(t_resampled, z_filt[N_fir_2:N+N_fir_2], label="Trial {}".format(trial_num))

# Se le da formato a los ejes de cada gráfica
    ax7[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    ax7[gesture_name][0].grid()
    ax7[gesture_name][0].legend(fontsize=6, loc='upper right');
    ax7[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    ax7[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    ax7[gesture_name][0].set_ylim(-6, 6)
    
    ax7[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    ax7[gesture_name][1].grid()
    ax7[gesture_name][1].legend(fontsize=6, loc='upper right');
    ax7[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    ax7[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    ax7[gesture_name][1].set_ylim(-6, 6)
    
    ax7[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    ax7[gesture_name][2].grid()
    ax7[gesture_name][2].legend(fontsize=6, loc='upper right');
    ax7[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    ax7[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)
    ax7[gesture_name][2].set_ylim(-6, 6)
    
    # fig7.savefig(f'{carpeta}\señales_filt_fir_windowed.png', dpi=200)

plt.show()

#%% Evaluar Performance del Filtro 

# Se aplica el filtrado sobre la señal 500 veces y se mide el tiempo requerido
# por el algoritmo
t_start_fir = time()
for i in range(500):
    senial_fir1 = signal.lfilter(Num_fir1, Den_fir1, senial)
t_end_fir = time()
t_start_iir = time()
for i in range(500):
    senial_fir2 = signal.lfilter(Num_fir2, Den_fir2, senial)
t_end_iir = time()

print("El algoritmo de filtrado FIR1 toma {:.3f}s".format(t_end_fir - t_start_fir))
print("El algoritmo de filtrado FIR2 toma {:.3f}s".format(t_end_iir - t_start_iir))

#Crear .h
process_code.fir_header('filtro_FIR_Windowed.h', Num_fir2)










