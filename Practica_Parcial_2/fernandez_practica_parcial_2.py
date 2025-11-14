# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:52:34 2025

@author: Aldana Fernandez
"""
"""
Enunciado: 
    Se desea implementar un sistema de filtrado que permita remover de una señal de ECG
artefactos producto del movimiento del paciente. La señal cuenta además con algunas
interferencias productos de armónicos superiores de la señal de línea (50Hz).

1. Proponga la función de transferencia analógica de un filtro pasa-banda con los
siguientes requerimientos:
● Atenuación menor a 0.1dB en la banda de paso (0.67 Hz a 40 Hz).
● Atenuación de al menos 30dB para frecuencias menores a 0.2 Hz, para
remover los artefactos de movimiento.
● Se desea utilizar la sección pasabajos del filtro pasa-banda como filtro anti
-alias. La señal será adquirida a una frecuencia de 300Hz con un CAD de 16
bits y Vref=5V.
● Cuenta con la señal ecg 860Hz.txt (muestreada a 860Hz) para analizar el
espectro y obtener los requisitos.


2. Factorice la H(s) obtenida en el punto anterior e implemente la sección pasa-bajos
de manera analógica calculando los componentes del circuito activo que la realice.
Utilice valores comerciales.
3. Simule el circuito activo obtenido en el punto anterior en LTSpice utilizando valores
comerciales de componentes y compare la respuesta en frecuencia obtenida con la
del sistema original. Indique si la considera adecuada.
4. Implemente la sección restante de manera digital.
5. Ensaye el filtro digital diseñado sobre la señal ecg_300Hz.txt, de manera de verificar
el correcto funcionamiento del filtro. Dicha señal ya ha sido filtrada con un filtro
pasa-bajos y remuestreada.
6. Diseñe un filtro FIR con PyFDA de similares características al obtenido a partir de la
factorización.
7. Grafique dos figuras presentando la versión filtrada superpuesta con la versión
original. Una figura para el dominio temporal y otra para el dominio frecuencial.


"""
#%%Importo las librerias que creo necesarias
from scipy import signal
#from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from funciones_fft import fft_mag
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
import filter_parameters
from matplotlib import patches

plt.close('all') # cerrar gráficas anteriores

#%% Item 1.

filename = 'ecg_860Hz.txt'          # nombre de archivo
senial_860 = np.loadtxt(filename)   # carga de la señal (se encuentra en V)
FS = 860                            # frecuencia de muestreo 860Hz

N= len(senial_860)                  # Cantidad de muestras = cantidad de datos en la señal
ts = 1 / FS                         # tiempo de muestreo
t = np.linspace(0, N * ts, N)       # vector de tiempos

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_title('Senial', fontsize=18)
ax.set_xlabel('Tiempo [s]', fontsize=15)
ax.set_ylabel('Magnitud [V]', fontsize=15)
ax.grid(True, which="both")
ax.plot(t, senial_860)
plt.show()

#%%Calculo la tranformada y la grafico

frec, fft_senial = fft_mag(senial_860, FS)

fig1, ax1 = plt.subplots(1, 1, figsize=(18, 12))
ax1.set_title('Tranformada de Fourier de la Senial', fontsize=18)
ax1.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1.set_ylabel('Magnitud [V]', fontsize=15)
ax1.grid(True, which="both")
ax1.plot(frec, fft_senial, label='FFT')
#ax1.set_xlim(0, 0.2)
plt.show()

#%%Como la condicion para el filtro es banda de paso (0.67 Hz a 40 Hz), 
#busco el maximo del espectro luego de 40 Hz

max_senial = 0
pos_max_senial = 0

# Determino que la banda de paso es hasta 40Hz asi que busco el maximo luego de ese valor
fb = 40
FS_resample = 300 
f_50 = 50
f_200 = 200
f_300 = 300
indices_validos = np.where(frec >= FS_resample/2)[0]

# Recorto el vector de busqueda, las magnitudes y frecuencias correspondientes
magnitudes_recortadas = fft_senial[indices_validos]
frecuencias_recortadas = frec[indices_validos]

# Buscar el índice del máximo dentro del subconjunto
indice_max = np.argmax(magnitudes_recortadas)
amp_max = magnitudes_recortadas[indice_max]
frecuencia_max = frecuencias_recortadas[indice_max]

indice = FS//2
amp_fs_2 = fft_senial[indice]
indice_50 = int(round(f_50 * N / FS))
indice_200 = int(round(f_200 * N / FS))
indice_300 = int(round(f_300 * N / FS))
amp_50 = fft_senial[indice_50]
amp_200 = fft_senial[indice_200]
amp_300 = fft_senial[indice_300]


print(f"Máximo valor: {amp_max} a {frecuencia_max} Hz")
print(f"Valor en fs/2: {amp_fs_2} a {FS/2} Hz")
print(f"Valor de {amp_50} a {f_50} Hz")
print(f"Valor de {amp_200} a {f_200} Hz")
print(f"Valor de {amp_300} a {f_300} Hz")


#%%Determino la resolucion del conversor 
#Datos del enunciado: La señal será adquirida a una frecuencia de 300Hz con un 
#CAD de 16 bits y Vref=5V.


# Parámetros ADC:
V_REF = 5        # Tensión de referencia en V
N_BITS = 16         # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en V

#%% Atenuaciones necesarias 
at_max = 20*np.log10(amp_max/RES) 
at_fs2_2 = 20*np.log10(amp_fs_2/RES)
at_50 = 20*np.log10(amp_50/RES)
at_200 = 20*np.log10(amp_200/RES)
at_300 = 20*np.log10(amp_300/RES)



print(f"Banda de paso hasta {fb} Hz")   # Banda de paso determinada en guía 1
print(f"Resolucion del conversor {RES}")
print(f"Atenuación mayor a {at_max:.2f}dB en {frecuencia_max}Hz")
print(f"Atenuación fs/2 a {at_fs2_2:.2f}dB en {FS/2}Hz")
print(f"Atenuación de {at_50:.2f}dB en {f_50}Hz")
print(f"Atenuación de {at_200:.2f}dB en {f_200}Hz")
print(f"Atenuación de {at_300:.2f}dB en {f_300}Hz")
print("\r")

#%% 3. Simule el circuito activo obtenido en el punto anterior en LTSpice utilizando valores
# comerciales de componentes y compare la respuesta en frecuencia obtenida con la
# del sistema original. Indique si la considera adecuada.

#Se planteo mal el problema, era solo el pasa bajo como analogico el pasa alto como digital

"""

f_1, mag_1 = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')

f_sim_1, mag_sim_1, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')

#fs2_2 = f[np.where(f>=(fs/2))][0]          # Valor más cercano a FS2/2
#f_fs_2 = f[ np.argmax( mag[ np.where(f == fs) ] ) ] + fs/2

fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
ax3.set_title('Filtro orden 6', fontsize=18)
ax3.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3.set_xscale('log')
ax3.grid(True, which="both")
ax3.plot(f_1,  mag_1, label='Diseñado')
ax3.plot(f_sim_1,  mag_sim_1, label='Simulado')
ax3.plot(FS/2, -at_fs2_2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(frecuencia_max, -at_max, marker='o', markersize=12, label='Requisito en máximo a partir de fs/2')
ax3.plot(f_50_Hz, -at_50_Hz, marker='X', markersize=12, label='Requisito en 50 Hz')
ax3.legend(loc="lower left", fontsize=15)

plt.show()

#Como el filtro anterior por los valores comerciales no cumple con la condicion de atenuacion maxima, planteo otro filtro

f_2, mag_2 = import_AnalogFilterWizard('DesignFiles_2/Data Files/Magnitude(dB).csv')

f_sim_2, mag_sim_2, _ = import_AC_LTSpice('DesignFiles_2/SPICE Files/LTspice/ACAnalysis.txt')

#fs2_2 = f[np.where(f>=(fs/2))][0]          # Valor más cercano a FS2/2
#f_fs_2 = f[ np.argmax( mag[ np.where(f == fs) ] ) ] + fs/2

fig4, ax4 = plt.subplots(1, 1, figsize=(12, 10))
ax4.set_title('Filtro orden 6', fontsize=18)
ax4.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax4.set_xscale('log')
ax4.grid(True, which="both")
ax4.plot(f_2,  mag_2, label='Diseñado')
ax4.plot(f_sim_2,  mag_sim_2, label='Simulado')
ax4.plot(FS/2, -at_fs2_2, marker='X', markersize=12, label='Requisito en fs/2')
ax4.plot(frecuencia_max, -at_max, marker='o', markersize=12, label='Requisito en máximo a partir de fs/2')
ax4.plot(f_50_Hz, -at_50_Hz, marker='X', markersize=12, label='Requisito en 50 Hz')
ax4.legend(loc="lower left", fontsize=15)
plt.show()

#Ahora con este filtro si cumplo con las condiciones
"""


#Filtro pasa bajo analogico 

f_3, mag_3 = import_AnalogFilterWizard('DesignFiles_pasa_bajo/Data Files/Magnitude(dB).csv')

f_sim_3, mag_sim_3, _ = import_AC_LTSpice('DesignFiles_pasa_bajo/SPICE Files/LTspice/ACAnalysis.txt')

#fs2_2 = f[np.where(f>=(fs/2))][0]          # Valor más cercano a FS2/2
#f_fs_2 = f[ np.argmax( mag[ np.where(f == fs) ] ) ] + fs/2

fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
ax3.set_title('Filtro orden 4', fontsize=18)
ax3.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3.set_ylabel('|H(jw)|² [dB]', fontsize=15)

ax3.set_xscale('log')
ax3.grid(True, which="both")
ax3.plot(f_3,  mag_3, label='Diseñado')
ax3.plot(f_sim_3,  mag_sim_3, label='Simulado')
ax3.plot(FS/2, -at_fs2_2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(frecuencia_max, -at_max, marker='o', markersize=12, label='Requisito en máximo a partir de fs/2')
ax3.plot(f_50, -at_50, marker='X', markersize=12, label='Requisito en 50 Hz')
ax3.plot(f_200, -at_200, marker='X', markersize=12, label='Requisito en 200 Hz')
ax3.plot(f_300, -at_300, marker='X', markersize=12, label='Requisito en 300 Hz')
#ax3.set_xlim(left=1e-1, right=500)
ax3.set_xlim( right=500)
ax3.legend(loc="lower left", fontsize=15)
plt.show()

#El filtro cumple con las condiciones

#%%4. Implemente la sección restante de manera digital.

FS_resample = 300                       # Frecuencia de muestreo para la cual estan diseñados los filtros
filename = 'ecg_300Hz.txt'              #Nombre del archivo donde esta la señal
senial_300 = np.loadtxt(filename)


ts_resample = 1 / FS_resample                     # tiempo de muestreo
N_resample = len(senial_300)                        # número de muestras en cada regsitro
t_resample = np.linspace(0, N_resample * ts_resample, N_resample)   # vector de tiempos


filtro_iir = np.load('IIR_buter_N5.npz', allow_pickle=True) 
filtro_fir = np.load('FIR.npz', allow_pickle=True) 
print("Filtro IIR: butterworth")
filter_parameters.filter_parameters('IIR_buter_N5.npz')
print("\r\n")
print("Filtro FIR: Kaiser")
filter_parameters.filter_parameters('FIR.npz')
print("\r\n")


# Se extraen los coeficientes de numerador y denominador
Num_iir, Den_iir = filtro_iir['ba'] 
Num_fir, Den_fir = filtro_fir['ba'] 

ceros_agregar = filtro_fir['N']//2 

print(f"La mitad del Orden del filtro FIR {ceros_agregar} ")

senial_fir = senial_300
senial_fir = np.pad(senial_300,(0,ceros_agregar),mode='constant')
senial_fir = signal.lfilter(Num_fir, Den_fir, senial_fir)
senial_fir = senial_fir[ceros_agregar:] 



senial_iir = signal.lfilter(Num_iir, Den_iir, senial_300)
f_300, senial_fft_mod_300 = fft_mag(senial_300, FS_resample)
frec_iir, rta_frec_iir = signal.freqz(Num_iir, Den_iir, worN=f_300, fs=FS_resample)
f1_iir, senial_iir_fft_mod = fft_mag(senial_iir, FS_resample)

frec_fir, rta_frec_fir = signal.freqz(Num_fir, Den_fir, worN=f_300, fs=FS_resample)
f1_fir, senial_fir_fft_mod = fft_mag(senial_fir, FS_resample)


#Grafica de la señal remuestrada a 300Hz

fig5, ax5 = plt.subplots(1, 1, figsize=(18, 12))
ax5.set_title('Senial con frecuencia de muestreo 300Hz', fontsize=18)
ax5.set_xlabel('Tiempo [s]', fontsize=15)
ax5.set_ylabel('Magnitud [V]', fontsize=15)
ax5.grid(True, which="both")
ax5.plot(t_resample, senial_300)
plt.show()

#grafica de la tranformada de los dos filtros digitales y de la señal a 300Hz

fig4, ax4 = plt.subplots(1, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Tranformada del filtro digital y de la señal", fontsize=18)
# Se grafican las respuestas de los filtros
ax4.plot(frec_fir, abs(rta_frec_fir), label='Filtro FIR', color='violet')
ax4.legend(loc="upper right", fontsize=15)

ax4.plot(frec_iir, abs(rta_frec_iir), label='Filtro IIR', color='green')
ax4.legend(loc="upper right", fontsize=15)

ax4.plot(f_300, senial_fft_mod_300/np.max(senial_fft_mod_300), label='Señal Original', color='blue')
ax4.set_ylabel('Magnitud (normalizada))', fontsize=15)
ax4.grid()
ax4.legend(loc="upper right", fontsize=15)
ax4.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4.legend(loc="upper right", fontsize=15)
#ax4.set_xlim([0, 300])
plt.show()


#%% 5. Ensaye el filtro digital diseñado sobre la señal ecg_300Hz.txt, de manera de verificar
#el correcto funcionamiento del filtro. Dicha señal ya ha sido filtrada con un filtro
#pasa-bajos y remuestreada.

fig6, ax6 = plt.subplots(1, 1, figsize=(15, 15), sharex=True)
fig6.suptitle("Tranformada de la señal original y la filtrada", fontsize=18)

# Se grafica la magnitud del espectro (normalizado)

# =============================================================================
# ax6.plot(f_300, senial_fft_mod_300/np.max(senial_fft_mod_300), label='Señal Original')
# ax6.set_ylabel('Magnitud (normalizada)', fontsize=15)
# ax6.set_xlabel('Frecuencia [Hz]', fontsize=15)
# ax6.grid()
# ax6.legend(loc="upper right", fontsize=15)
# ax6.plot(f1_fir, senial_fir_fft_mod/np.max(senial_fft_mod_300), 
#            label='Senial Filtrada FIR', color='violet')
# ax6.legend(loc="upper right", fontsize=15)
# ax6.plot(f1_iir, senial_iir_fft_mod/np.max(senial_fft_mod_300), 
#             label='Senial Filtrada IIR', color='green')
# ax6.legend(loc="upper right", fontsize=15)
# #ax6.set_xlim([0, 20])
# plt.show()
# =============================================================================


# Se grafica la magnitud del espectro (sin normalizar)

ax6.plot(f_300, senial_fft_mod_300, label='Señal Original')
ax6.set_ylabel('Magnitud (normalizada)', fontsize=15)
ax6.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax6.grid()
ax6.legend(loc="upper right", fontsize=15)
ax6.plot(f1_fir, senial_fir_fft_mod, 
           label='Senial Filtrada FIR', color='violet')
ax6.legend(loc="upper right", fontsize=15)
ax6.plot(f1_iir, senial_iir_fft_mod, 
            label='Senial Filtrada IIR', color='green')
ax6.legend(loc="upper right", fontsize=15)
#ax6.set_xlim([0, 20])
plt.show()



#%% Grafica de la señal original y las filtradas digitalmente
# las tres graficas en un solo grafico
fig7, ax7 = plt.subplots(1, 1, figsize=(15, 15), sharex=True)
fig7.suptitle("Señal original y la filtrada", fontsize=18)
ax7.plot(t_resample, senial_300, label='Señal Original')
ax7.set_ylabel('Amplitud [V]', fontsize=15)
ax7.set_xlabel('Tiempo [s]', fontsize=15)
ax7.plot(t_resample, senial_iir, label='Señal filtrada IIR', color = 'green')
ax7.plot(t_resample, senial_fir, label='Señal filtrada FIR', color = 'violet')
ax7.grid()
ax7.legend(loc="upper right", fontsize=15)
plt.show()

# Una grafica con el filtro iir y otra con el fir
fig8, ax8 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig8.suptitle("Señal original y la filtrada", fontsize=18)
ax8[0].set_title('Aplico Filtro FIR', fontsize=15)
ax8[0].plot(t_resample, senial_300, label='Señal Original', color = 'tab:blue')
ax8[0].set_ylabel('Amplitud [V]', fontsize=15)
ax8[0].set_xlabel('Tiempo [s]', fontsize=15)
ax8[0].plot(t_resample, senial_iir, label='Señal filtrada IIR', color = 'green')

ax8[0].grid()
ax8[0].legend(loc="upper right", fontsize=15)

ax8[1].set_title('Aplico Filtro IIR', fontsize=15)
ax8[1].plot(t_resample, senial_fir, label='Señal filtrada FIR', color = 'violet')
ax8[1].plot(t_resample, senial_300, label='Señal Original', color = 'tab:blue')
ax8[1].set_ylabel('Amplitud [V]', fontsize=15)
ax8[1].set_xlabel('Tiempo [s]', fontsize=15)
ax8[1].grid()
ax8[1].legend(loc="upper right", fontsize=15)
plt.show()



#%% Graficar los ceros y los polos de los filtros: 


# Se extraen polos y ceros de los filtros
zeros_fir, polos_fir, k_fir =   filtro_fir['zpk']  
zeros_iir, polos_iir, k_iir =   filtro_iir['zpk']

# Se grafican las distribuciones de ceros y polos
fig9, ax9 = plt.subplots(1, 2, figsize=(15, 7))
fig9.suptitle("Distribución de Ceros y Polos en el plano Z", fontsize=18)

ax9[0].set_title('Filtro FIR', fontsize=15)
ax9[0].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax9[0].plot(polos_fir.real, polos_fir.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax9[0].plot(zeros_fir.real, zeros_fir.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_fir)), np.max(abs(zeros_fir))])
ax9[0].set_xlim(-lim, lim)
ax9[0].set_ylim(-lim, lim)
ax9[0].set_ylabel('Imag(z)', fontsize=15)
ax9[0].set_xlabel('Real(z)', fontsize=15)
ax9[0].grid()
ax9[0].legend(loc="upper right", fontsize=12)

ax9[1].set_title('Filto IIR', fontsize=15)
ax9[1].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax9[1].plot(polos_iir.real, polos_iir.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax9[1].plot(zeros_iir.real, zeros_iir.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_iir)), np.max(abs(zeros_iir))])
ax9[1].set_xlim(-lim, lim)
ax9[1].set_ylim(-lim, lim)
ax9[1].set_ylabel('Imag(z)', fontsize=15)
ax9[1].set_xlabel('Real(z)', fontsize=15)
ax9[1].grid()
ax9[1].legend(loc="upper right", fontsize=12)


plt.show()








