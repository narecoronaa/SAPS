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

FS = 500 # Frecuencia de muestreo: 500Hz
T = 3    # Tiempo total de cada registro: 3 segundos

folder = 'dataset_voley' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)
print("\r\n")

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada registro
t = np.linspace(0, N * ts, N)   # vector de tiempos

#%% Cálculo y Graficación de la Transformada de Fourier

IIR1 ='Filtrado IIR - Chebyshev'
IIR2 ='Filtrado IIR - Butterworth'
carpeta = r'C:\Users\narec\Documents\SAPS\TP\Filtros_Digitales_Graficas'

# Parámetros para el remuestreo de las señales
FS_resample = 60                        # Frecuencia de muestreo para la cual están diseñados los filtros
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
ax1[0].set_title(IIR1, fontsize=15)
ax1[1].plot(t_resampled, senial, label='Señal Contaminada')
ax1[1].set_ylabel('Tensión [V]', fontsize=15)
ax1[1].grid()
ax1[1].legend(loc="upper right", fontsize=15)
ax1[1].set_xlabel('Tiempo [s]', fontsize=15)
ax1[1].set_xlim([0, ts*N])
ax1[1].set_title(IIR2, fontsize=15)

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
ax2[0].set_title(IIR1, fontsize=15)
ax2[1].plot(f, senial_fft_mod, label='Señal Original')
ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[1].grid()
ax2[1].legend(loc="upper right", fontsize=15)
ax2[1].set_title(IIR2, fontsize=15)
ax2[1].set_ylabel('Magnitud', fontsize=15)
ax2[1].set_xlim([0, FS_resample/2])

#%% Carga de los Filtros 

# Se cargan los archivos generados mediante pyFDA
filtro_iir1 = np.load('IIR_chebyshev_O4.npz', allow_pickle=True)
filtro_iir2 = np.load('IIR_butterworth_O4.npz', allow_pickle=True) 

# Se muestran parámetros de diseño
print("FILTRO IIR 1: " + IIR1)
filter_parameters.filter_parameters('IIR_chebyshev_O4.npz')
print("\r\n")
print("FILTRO IIR 2: " + IIR2)
filter_parameters.filter_parameters('IIR_butterworth_O4.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_iir1, Den_iir1 = filtro_iir1['ba']     
Num_iir2, Den_iir2 = filtro_iir2['ba'] 

# Se expresan las funciones de transferencias (H(z))
z_sym = sy.Symbol('z') # Se crea una variable simbólica z
Hz = sy.Symbol('H(z)')

# Función de transferencia IIR1
Numz_iir1 = 0
Denz_iir1 = 0
for i in range(len(Num_iir1)):
    Numz_iir1 += Num_iir1[i] * np.power(z_sym, -i)
for i in range(len(Den_iir1)):
    Denz_iir1 += Den_iir1[i] * np.power(z_sym, -i)
#print(f"La función de transferencia del {IIR1} es:")
#print(sy.pretty(sy.Eq(Hz, Numz_iir1.evalf(3) / Denz_iir1.evalf(3)))) 
#print("\r\n")

# Función de transferencia IIR2
Numz_iir2 = 0
Denz_iir2 = 0
for i in range(len(Num_iir2)):
    Numz_iir2 += Num_iir2[i] * np.power(z_sym, -i)
for i in range(len(Den_iir2)):
    Denz_iir2 += Den_iir2[i] * np.power(z_sym, -i)
#print(f"La función de transferencia del {IIR2} es:")
#print(sy.pretty(sy.Eq(Hz, Numz_iir2.evalf(3) / Denz_iir2.evalf(3)))) 
#print("\r\n")

#%% Análisis de los Filtros 
f_log = np.logspace(-1, 2, int(10e3))  # vector de frecuencia
# Se calcula la respuesta en frecuencia de los filtros
f_iir1, h_iir1 = signal.freqz(Num_iir1, Den_iir1, worN=f_log, fs=FS_resample)
f_iir2, h_iir2 = signal.freqz(Num_iir2, Den_iir2, worN=f_log, fs=FS_resample)

# Se crea una gráfica 
fig3, ax3 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig3.suptitle("Respuesta en Frecuencia de los filtros", fontsize=18)

# Se grafican las respuestas de los filtros
ax3[0].plot(f_iir1, abs(h_iir1), label=IIR1, color='orange')
ax3[0].legend(loc="upper right", fontsize=15)
ax3[0].set_title(IIR1, fontsize=15)
ax3[0].set_xlim([0, FS_resample/2])
ax3[0].grid()

ax3[1].plot(f_iir2, abs(h_iir2), label=IIR2, color='green')
ax3[1].legend(loc="upper right", fontsize=15)
ax3[1].set_title(IIR2, fontsize=15)
ax3[1].set_xlim([0, FS_resample/2])
ax3[1].grid()

#RESPUESTA IMPLEMENTADA
# Datos de frecuencia y salida (en G)
f_impl = [0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 5, 10, 11, 12, 13, 14, 15, 16, 17, 17.5, 18, 20, 21, 23, 25, 28, 29]
salida_impl = [7, 7.1, 7, 7, 7, 7, 7, 7, 8, 8, 7, 5, 6, 6.8, 7, 7, 7, 6.5, 6, 6, 6, 3.5, 3.5, 2.2, 1.2, 0.2, 0.05]

# Ganancia en dB respecto a 7G
#ganancia_impl_db = [20 * np.log10(y/7) for y in salida_impl]
ganancia_impl = [y/7 for y in salida_impl]

# Graficar la respuesta implementada sobre el Chebyshev
ax3[0].plot(f_impl, ganancia_impl, 'o-', color='blue', label='Implementado')


#fig3.savefig(rf'{carpeta}\respuesta_en_frec_iir.png', dpi=200)

# Se evalúa la atenuación en las frecuencias de interés 
_, h1_iir1 = signal.freqz(Num_iir1, Den_iir1, worN=[0.01, 20], fs=FS_resample)
_, h1_iir2 = signal.freqz(Num_iir2, Den_iir2, worN=[0.01, 20], fs=FS_resample)

#print(f"La atenuación del {IIR1} en 0.01Hz es de {20*np.log10(abs(h1_iir1[0])):.2f}dB")
#print(f"La atenuación del {IIR1} en 20Hz es de {20*np.log10(abs(h1_iir1[1])):.2f}dB")
#print(f"La atenuación del {IIR2} en 0.01Hz es de {20*np.log10(abs(h1_iir2[0])):.2f}dB")
#print(f"La atenuación del {IIR2} en 20Hz es de {20*np.log10(abs(h1_iir2[1])):.2f}dB")
#print("\r\n")

# Se extraen polos y ceros de los filtros
zeros_iir1, polos_iir1, k_iir1 = filtro_iir1['zpk']  
zeros_iir2, polos_iir2, k_iir2 = filtro_iir2['zpk']

# Se grafican las distribuciones de ceros y polos
fig4, ax4 = plt.subplots(1, 2, figsize=(15, 7))
fig4.suptitle("Distribución de Ceros y Polos en el plano Z", fontsize=18)

ax4[0].set_title(IIR1, fontsize=15)
ax4[0].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax4[0].plot(polos_iir1.real, polos_iir1.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax4[0].plot(zeros_iir1.real, zeros_iir1.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_iir1)), np.max(abs(zeros_iir1))])
ax4[0].set_xlim(-lim, lim)
ax4[0].set_ylim(-lim, lim)
ax4[0].set_ylabel('Imag(z)', fontsize=15)
ax4[0].set_xlabel('Real(z)', fontsize=15)
ax4[0].grid()
ax4[0].legend(loc="upper right", fontsize=12)

ax4[1].set_title(IIR2, fontsize=15)
ax4[1].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax4[1].plot(polos_iir2.real, polos_iir2.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax4[1].plot(zeros_iir2.real, zeros_iir2.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_iir2)), np.max(abs(zeros_iir2))])
ax4[1].set_xlim(-lim, lim)
ax4[1].set_ylim(-lim, lim)
ax4[1].set_ylabel('Imag(z)', fontsize=15)
ax4[1].set_xlabel('Real(z)', fontsize=15)
ax4[1].grid()
ax4[1].legend(loc="upper right", fontsize=12)

#fig4.savefig(f'{carpeta}\ceros_polos_iir.png', dpi=200)
#%% Filtrado de la Señal 

# Se aplica el filtrado sobre la señal
senial_iir1 = signal.lfilter(Num_iir1, Den_iir1, senial)
senial_iir2 = signal.lfilter(Num_iir2, Den_iir2, senial)

# Se grafican las señales filtradas
ax1[0].plot(t_resampled, senial_iir1, label='Señal Filtrada (IIR1)', color='red')
ax1[0].legend(loc="upper right", fontsize=15)
ax1[1].plot(t_resampled, senial_iir2, label='Señal Filtrada (IIR2)', color='purple')
ax1[1].legend(loc="upper right", fontsize=15)

# Se calculan y grafican sus espectros (normalizados)
f1_iir1, senial_iir1_fft_mod = funciones_fft.fft_mag(senial_iir1, FS_resample)
f1_iir2, senial_iir2_fft_mod = funciones_fft.fft_mag(senial_iir2, FS_resample)
ax2[0].plot(f1_iir1, senial_iir1_fft_mod, 
            label='Senial Filtrada IIR1', color='red')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f1_iir2, senial_iir2_fft_mod, 
            label='Senial Filtrada IIR2', color='purple')
ax2[1].legend(loc="upper right", fontsize=15)

#fig1.savefig(rf'{carpeta}\aceleracion_contaminada_iir.png', dpi=200)
#fig2.savefig(rf'{carpeta}\aceleracion_original_iir.png', dpi=200)
plt.show()

#%% Graficación señales sin filtrar

fig5, ax5 = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig5.subplots_adjust(hspace=0.5)
fig5.suptitle("Señales sin filtrar", fontsize=15)

trial_num = 0
for gesture_name in classmap:                           
    for capture in range(int(len(x))):                  
        if (x[capture, N] == gesture_name):             
            ax5[gesture_name][0].plot(t_resampled, signal.resample(x[capture, 0:N], N_resample), label="Trial {}".format(trial_num))
            ax5[gesture_name][1].plot(t_resampled, signal.resample(y[capture, 0:N], N_resample), label="Trial {}".format(trial_num))
            ax5[gesture_name][2].plot(t_resampled, signal.resample(z[capture, 0:N], N_resample), label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

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

plt.show()

#%% Graficación señales filtradas (IIR1)

fig6, ax6 = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig6.subplots_adjust(hspace=0.5)
fig6.suptitle("Señales filtradas con filtro IIR - CHEBYSHEV", fontsize=15)

trial_num = 0
for gesture_name in classmap:                           
    for capture in range(int(len(x))):                  
        if (x[capture, N] == gesture_name):             
            x_filt = signal.lfilter(Num_iir1, Den_iir1, signal.resample(x[capture, 0:N], N_resample))
            y_filt = signal.lfilter(Num_iir1, Den_iir1, signal.resample(y[capture, 0:N], N_resample))
            z_filt = signal.lfilter(Num_iir1, Den_iir1, signal.resample(z[capture, 0:N], N_resample))
            ax6[gesture_name][0].plot(t_resampled, x_filt, label="Trial {}".format(trial_num))
            ax6[gesture_name][1].plot(t_resampled, y_filt, label="Trial {}".format(trial_num))
            ax6[gesture_name][2].plot(t_resampled, z_filt, label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

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
    
    #fig6.savefig(f'{carpeta}\señales_filt_iir_chebyshev.png', dpi=200)

plt.show()

#%% Graficación señales filtradas (IIR2)

fig7, ax7 = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig7.subplots_adjust(hspace=0.5)
fig7.suptitle("Señales filtradas con filtro IIR - BUTTERWORTH", fontsize=15)

trial_num = 0
for gesture_name in classmap:                           
    for capture in range(int(len(x))):                  
        if (x[capture, N] == gesture_name):             
            x_filt = signal.lfilter(Num_iir2, Den_iir2, signal.resample(x[capture, 0:N], N_resample))
            y_filt = signal.lfilter(Num_iir2, Den_iir2, signal.resample(y[capture, 0:N], N_resample))
            z_filt = signal.lfilter(Num_iir2, Den_iir2, signal.resample(z[capture, 0:N], N_resample))
            ax7[gesture_name][0].plot(t_resampled, x_filt, label="Trial {}".format(trial_num))
            ax7[gesture_name][1].plot(t_resampled, y_filt, label="Trial {}".format(trial_num))
            ax7[gesture_name][2].plot(t_resampled, z_filt, label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

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
    
    #fig7.savefig(f'{carpeta}\señales_filt_iir_butterworth.png', dpi=200)

plt.show()

#%% Evaluar Performance del Filtro 

# Se aplica el filtrado sobre la señal 500 veces y se mide el tiempo requerido por el algoritmo
t_start_iir1 = time()
for i in range(500):
    senial_iir1 = signal.lfilter(Num_iir1, Den_iir1, senial)
t_end_iir1 = time()
t_start_iir2 = time()
for i in range(500):
    senial_iir2 = signal.lfilter(Num_iir2, Den_iir2, senial)
t_end_iir2 = time()

print("El algoritmo de filtrado IIR1 toma {:.3f}s".format(t_end_iir1 - t_start_iir1))
print("El algoritmo de filtrado IIR2 toma {:.3f}s".format(t_end_iir2 - t_start_iir2))


#Crear .h
#process_code.iir_sos_header('filtro_IIR_Chebyshev.h', signal.tf2sos(Num_iir1, Den_iir1))


