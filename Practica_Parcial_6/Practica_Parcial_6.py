# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:01:10 2025

Datos del enunciado: 
                    CAD de 14 bits (Vref: 3.3V) muestreando a 15 Hz,
para luego separar las dos señales digitalmente. Para ello se le pide:
                    fs = 1k Hz despues se muestrea a 15 Hz
                    Los archivos estan mV

@author: Equipo
"""
#%% Include necesasrios 

import numpy as np
import matplotlib.pyplot as plt
from funciones_fft import fft_mag
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
import filter_parameters
from scipy import signal
#from scipy.io import wavfile
#from matplotlib import patches
#from scipy.interpolate import interp1d



#%% 1. Levanto la seniales y las grafico tanto en el tiempo como en la frecuencia 


archivo_1 = 'seniales/pleth_72lpm_1000hz.txt' # nombre de archivo
archivo_2 = 'seniales/pleth_90lpm_1000hz.txt' # nombre de archivo
archivo_3 = 'seniales/pleth_120lpm_1000hz.txt' # nombre de archivo

senial_1 = np.loadtxt(archivo_1) # la amplitud de la señal se encuentra en mV
senial_2 = np.loadtxt(archivo_2) # la amplitud de la señal se encuentra en mV
senial_3 = np.loadtxt(archivo_3) # la amplitud de la señal se encuentra en mV

fs = 1000 # frecuencia de muestreo 1000Hz

ts = 1 / fs # tiempo de muestreo

N_1 = len(senial_1) # número de muestras en el archivo de audio
N_2 = len(senial_2) # número de muestras en el archivo de audio
N_3 = len(senial_3) # número de muestras en el archivo de audio

t_1 = np.linspace(0, N_1 * ts, N_1) # vector de tiempo -> la función la da x espaciados
t_2 = np.linspace(0, N_2 * ts, N_2) # vector de tiempo -> la función la da x espaciados
t_3 = np.linspace(0, N_3 * ts, N_3) # vector de tiempo -> la función la da x espaciados


# Grafico las seniales en el tiempo
fig1, ax1 = plt.subplots(3, 1, figsize=(20, 20))
fig1.suptitle("Seniales en el tiempo", fontsize=20)
ax1[0].set_xlabel('Tiempo [s]', fontsize=15)
ax1[0].set_ylabel('Magnitud [mV]', fontsize=15)
ax1[0].grid(True, which="both")
ax1[0].plot(t_1, senial_1, color = 'blue')

ax1[1].set_xlabel('Tiempo [s]', fontsize=15)
ax1[1].set_ylabel('Magnitud [mV]', fontsize=15)
ax1[1].grid(True, which="both")
ax1[1].plot(t_2, senial_2, color = 'purple')

ax1[2].set_xlabel('Tiempo [s]', fontsize=15)
ax1[2].set_ylabel('Magnitud [mV]', fontsize=15)
ax1[2].grid(True, which="both")
ax1[2].plot(t_3, senial_3, color = 'magenta')
plt.show()

#Calculo las tranformadas de las seniales
frec_1, fft_senial_1 = fft_mag(senial_1, fs)
frec_2, fft_senial_2 = fft_mag(senial_2, fs)
frec_3, fft_senial_3 = fft_mag(senial_3, fs)

# Grafico las seniales en frecuencia
fig2, ax2 = plt.subplots(3, 1, figsize=(20, 20))
fig2.suptitle("Seniales en la frecuencia", fontsize=20)
ax2[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[0].set_ylabel('Magnitud [mV]', fontsize=15)
ax2[0].grid(True, which="both")
ax2[0].plot(frec_1, fft_senial_1, color = 'blue')

ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[1].set_ylabel('Magnitud [mV]', fontsize=15)
ax2[1].grid(True, which="both")
ax2[1].plot(frec_2, fft_senial_2, color = 'purple')

ax2[2].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[2].set_ylabel('Magnitud [mV]', fontsize=15)
ax2[2].grid(True, which="both")
ax2[2].plot(frec_3, fft_senial_3, color = 'magenta')

plt.show()


#Para buscar las caracteristicas del filtro antialias, tengo en cuenta que 
# despues se va a muestrear a 15Hz, busco los maximos luego de 7,5 Hz


#Determino los indices validos (los que corresponden a frecuencias mayores a FS/2) 
FS_resample = 15 
indices_validos_1 = np.where(frec_1 >= FS_resample/2)[0]
indices_validos_2 = np.where(frec_2 >= FS_resample/2)[0]
indices_validos_3 = np.where(frec_3 >= FS_resample/2)[0]

#Recorto el vector de busqueda, las magnitudes y frecuencias correspondientes
magnitudes_recortadas_1 = fft_senial_1[indices_validos_1]
magnitudes_recortadas_2 = fft_senial_2[indices_validos_2]
magnitudes_recortadas_3 = fft_senial_3[indices_validos_3]


frecuencias_recortadas_1 = frec_1[indices_validos_1]
frecuencias_recortadas_2 = frec_2[indices_validos_2]
frecuencias_recortadas_3 = frec_3[indices_validos_3]

# Buscar el índice del máximo dentro de cada subconjunto
indice_max_1 = np.argmax(magnitudes_recortadas_1)
indice_max_2 = np.argmax(magnitudes_recortadas_2)
indice_max_3 = np.argmax(magnitudes_recortadas_3)

# Con el indice busco a que amplitud corresponde en cada seniañ
amp_max_1 = magnitudes_recortadas_1[indice_max_1]
amp_max_2 = magnitudes_recortadas_2[indice_max_2]
amp_max_3 = magnitudes_recortadas_3[indice_max_3]

frecuencia_max_1 = frecuencias_recortadas_1[indice_max_1]
frecuencia_max_2 = frecuencias_recortadas_2[indice_max_2]
frecuencia_max_3 = frecuencias_recortadas_3[indice_max_3]


fs2_1 = frec_1[np.where(frec_1>=(FS_resample/2))][0] #valor mas cercano a la fm/2 ene l vector de frecuencias
fs2_2 = frec_2[np.where(frec_2>=(FS_resample/2))][0] #valor mas cercano a la fm/2 ene l vector de frecuencias
fs2_3 = frec_3[np.where(frec_3>=(FS_resample/2))][0] #valor mas cercano a la fm/2 ene l vector de frecuencias

amp_fm_2_1 = np.max( fft_senial_1 [np.where(frec_1 == fs2_1) ] )
amp_fm_2_2 = np.max( fft_senial_2 [np.where(frec_2 == fs2_2) ] )
amp_fm_2_3 = np.max( fft_senial_3 [np.where(frec_3 == fs2_3) ] )

frec_fm_2_1 = frec_1[np.argmax(fft_senial_1[np.where(frec_1 == fs2_1) ] ) ] + fs2_1
frec_fm_2_2 = frec_2[np.argmax(fft_senial_2[np.where(frec_2 == fs2_2) ] ) ] + fs2_2
frec_fm_2_3 = frec_1[np.argmax(fft_senial_1[np.where(frec_3 == fs2_3) ] ) ] + fs2_3


print("En la senial 1")
print(f"Máximo valor: {amp_max_1} a {frecuencia_max_1} Hz")
print(f"Valor en fs/2: {amp_fm_2_1} a {frec_fm_2_1} Hz \n")

print("En la senial 2")
print(f"Máximo valor: {amp_max_2} a {frecuencia_max_2} Hz")
print(f"Valor en fs/2: {amp_fm_2_2} a {frec_fm_2_2} Hz \n")

print("En la senial 3")
print(f"Máximo valor: {amp_max_3} a {frecuencia_max_3} Hz")
print(f"Valor en fs/2: {amp_fm_2_3} a {frec_fm_2_3} Hz \n")

frec_max = 0

#Busco el maximo de las fs/2
max_fs2 = 0
if(max_fs2 < amp_fm_2_1):
    max_fs2 = amp_fm_2_1

if(max_fs2 < amp_fm_2_2):
    max_fs2 = amp_fm_2_2

if(max_fs2 < amp_fm_2_3):
    max_fs2 = amp_fm_2_3
    
    


#Ahora calculo la atenuacion que necesito cumplir con el filtro antialias
V_REF = 3300                     # Tensión de referencia en mV
N_BITS = 14                     # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en V

at_fs2_max = 20*np.log10(max_fs2/RES)

at_fs2_1 = 20*np.log10(amp_fm_2_1/RES)
at_fs2_2 = 20*np.log10(amp_fm_2_2/RES)
at_fs2_3 = 20*np.log10(amp_fm_2_3/RES)

at_1 =  20*np.log10(amp_max_1/RES) 
at_2 =  20*np.log10(amp_max_2/RES) 
at_3 =  20*np.log10(amp_max_3/RES) 

at_max = 0
 
if(at_max < at_1):
    at_max = at_1
    frec_at_max = frecuencia_max_1 

if(at_max < at_2):
    at_max = at_2
    frec_at_max = frecuencia_max_2 

if(at_max < at_3):
    at_max = at_3
    frec_at_max = frecuencia_max_3 

print(f"Resolucion del conversor {RES} \n")

#Muestro todas las atenuaciones.

print("Los requisitos para el filtro antialisa son: ")
print(f"Atenuación maxima en senial 1: {at_1:.2f}dB en {frecuencia_max_1:.3f}Hz")
print(f"Atenuación maxima en senial 2: {at_2:.2f}dB en {frecuencia_max_2:.3f}Hz")
print(f"Atenuación maxima en senial 3: {at_3:.2f}dB en {frecuencia_max_3:.3f}Hz \n")

print(f"Atenuación fs/2 a {at_fs2_1:.2f}dB en {frec_fm_2_1}Hz")
print(f"Atenuación fs/2 a {at_fs2_2:.2f}dB en {frec_fm_2_2}Hz")
print(f"Atenuación fs/2 a {at_fs2_3:.2f}dB en {frec_fm_2_3}Hz\n")

print(f"Atenuación max en fs/2 a {at_fs2_max:.2f}dB en {frec_fm_2_3}Hz")
print(f"Atenuación max despues de fs/2 es {at_max:.2f}dB en {frecuencia_max_1}Hz") 

#Grafico la respuesta del filtro diseñado y el simulado con las restricciones de diseño

#Grafico el filtro diseñado y el simulado con las condiciones de diseño 
# para ver si cumpleo no

frec_dis, mag_dis = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')
frec_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')


# Análisis de la atenuación del filtro simulado en las frecuencias de interés
F_AT1 = FS_resample/2
F_AT2 = frec_at_max
# se calcula la atenuación en el punto mas cercano a la frecuencia de interés
at1 = mag_sim[np.argmin(np.abs(frec_sim-F_AT1))] 
at2 = mag_sim[np.argmin(np.abs(frec_sim-F_AT2))]

print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(F_AT1, at1))
print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(F_AT2, at2))
print("\r")


fig3, ax3 = plt.subplots(1, 1, figsize=(22, 15))
ax3.set_title('Filtro orden 4 Butterworth', fontsize=18)
ax3.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3.set_xscale('log')
ax3.grid(True, which="both")
ax3.plot(frec_dis,  mag_dis, label='Diseñado')
ax3.plot(frec_sim,  mag_sim, label='Simulado')
ax3.plot(FS_resample/2, -at_fs2_max, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(frec_at_max, -at_max, marker='X', markersize=12, label='Requisito en máximo')
ax3.legend(loc="lower left", fontsize=15)
plt.show()

#%% 3. Se diseño el filtro digital iir pasabanda tipo Butterworth, con las condiciones dadas 

#%% 4. Se diseño el filtro digital fir pasabanda kaiser, con las condiciones dadas 

#%% 5. Probar el funcionamiento de ambos filtros utilizando las señales de prueba muestreadas a 15
# Hz (pleth_72lpm_15hz.txt, pleth_90lpm_15hz.txt, pleth_120lpm_15hz.txt). Graficar las
# señales antes y después de ser filtradas.

#Cargo las nuevas seniales. 

archivo_1_rem = 'seniales/pleth_72lpm_15hz.txt' # nombre de archivo
archivo_2_rem = 'seniales/pleth_90lpm_15hz.txt' # nombre de archivo
archivo_3_rem = 'seniales/pleth_120lpm_15hz.txt' # nombre de archivo

senial_1_rem = np.loadtxt(archivo_1_rem) # la amplitud de la señal se encuentra en mV
senial_2_rem = np.loadtxt(archivo_2_rem) # la amplitud de la señal se encuentra en mV
senial_3_rem = np.loadtxt(archivo_3_rem) # la amplitud de la señal se encuentra en mV

ts_rem = 1 / FS_resample # tiempo de muestreo
N_1_rem = len(senial_1_rem) # número de muestras en el archivo de audio
t_1_rem = np.linspace(0, N_1_rem * ts_rem, N_1_rem) # vector de tiempo -> la función la da x espaciados

N_2_rem = len(senial_2_rem) # número de muestras en el archivo de audio
t_2_rem = np.linspace(0, N_2_rem * ts_rem, N_2_rem) # vector de tiempo -> la función la da x espaciados

N_3_rem = len(senial_3_rem) # número de muestras en el archivo de audio
t_3_rem = np.linspace(0, N_3_rem * ts_rem, N_3_rem) # vector de tiempo -> la función la da x espaciados


#Primero aplico el filtro iir a las 3 seniales y despues el filtro fir 

filtro_iir = np.load('filtros_digitales/filtro_iir.npz', allow_pickle=True) 
print("Filtro IIR: butterworth")
filter_parameters.filter_parameters('filtros_digitales/filtro_iir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_iir, Den_iir = filtro_iir['ba'] 


senial_iir_1 = signal.lfilter(Num_iir, Den_iir, senial_1_rem)
senial_iir_2 = signal.lfilter(Num_iir, Den_iir, senial_2_rem)
senial_iir_3 = signal.lfilter(Num_iir, Den_iir, senial_3_rem)


filtro_fir = np.load('filtros_digitales/filtro_fir.npz', allow_pickle=True) 
print("Filtro FIR: kaiser")
filter_parameters.filter_parameters('filtros_digitales/filtro_fir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_fir, Den_fir = filtro_fir['ba'] 

ceros_agregar = filtro_fir['N']//2 

print(f"La mitad del Orden del filtro FIR {ceros_agregar} ")

senial_fir_1 = senial_1_rem
senial_fir_2 = senial_2_rem
senial_fir_3 = senial_3_rem

senial_fir_1 = np.pad(senial_1_rem,(0,ceros_agregar),mode='constant')
senial_fir_2 = np.pad(senial_2_rem,(0,ceros_agregar),mode='constant')
senial_fir_3 = np.pad(senial_3_rem,(0,ceros_agregar),mode='constant')

senial_fir_1 = signal.lfilter(Num_fir, Den_fir, senial_fir_1)
senial_fir_2 = signal.lfilter(Num_fir, Den_fir, senial_fir_2)
senial_fir_3 = signal.lfilter(Num_fir, Den_fir, senial_fir_3)

senial_fir_1 = senial_fir_1[ceros_agregar:] 
senial_fir_2 = senial_fir_2[ceros_agregar:] 
senial_fir_3 = senial_fir_3[ceros_agregar:] 

fig4, ax4 = plt.subplots(3, 1, figsize=(22, 15), sharex=True)
fig4.suptitle("Senial 1 remuestreada y las filtradas", fontsize=18)
ax4[0].plot(t_1_rem, senial_1_rem, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[0].set_xlabel('Tiempo [s]', fontsize=15)
ax4[0].grid()
ax4[1].plot(t_1_rem, senial_iir_1, label='Señal filtrada iir', color='blue')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[1].set_xlabel('Tiempo [s]', fontsize=15)
ax4[1].grid()
ax4[2].plot(t_1_rem, senial_fir_1, label='Señal filtrada fir', color='blue')
ax4[2].legend(loc="upper right", fontsize=15)
ax4[2].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[2].set_xlabel('Tiempo [s]', fontsize=15)
ax4[2].grid()

plt.show()

fig5, ax5 = plt.subplots(3, 1, figsize=(22, 15), sharex=True)
fig5.suptitle("Senial 2 remuestreada y las filtradas", fontsize=18)
ax5[0].plot(t_2_rem, senial_2_rem, label='Señal Original', color='purple')
ax5[0].legend(loc="upper right", fontsize=15)
ax5[0].set_ylabel('Magnitud [mV])', fontsize=15)
ax5[0].set_xlabel('Tiempo [s]', fontsize=15)
ax5[0].grid()
ax5[1].plot(t_2_rem, senial_iir_2, label='Señal filtrada iir', color='purple')
ax5[1].legend(loc="upper right", fontsize=15)
ax5[1].set_ylabel('Magnitud [mV])', fontsize=15)
ax5[1].set_xlabel('Tiempo [s]', fontsize=15)
ax5[1].grid()
ax5[2].plot(t_2_rem, senial_fir_2, label='Señal filtrada fir', color='purple')
ax5[2].legend(loc="upper right", fontsize=15)
ax5[2].set_ylabel('Magnitud [mV])', fontsize=15)
ax5[2].set_xlabel('Tiempo [s]', fontsize=15)
ax5[2].grid()

plt.show()

fig6, ax6 = plt.subplots(3, 1, figsize=(22, 15), sharex=True)
fig6.suptitle("Senial 3 remuestreada y las filtradas", fontsize=18)
ax6[0].plot(t_3_rem, senial_3_rem, label='Señal Original', color='magenta')
ax6[0].legend(loc="upper right", fontsize=15)
ax6[0].set_ylabel('Magnitud [mV])', fontsize=15)
ax6[0].set_xlabel('Tiempo [s]', fontsize=15)
ax6[0].grid()
ax6[1].plot(t_3_rem, senial_iir_3, label='Señal filtrada iir', color='magenta')
ax6[1].legend(loc="upper right", fontsize=15)
ax6[1].set_ylabel('Magnitud [mV])', fontsize=15)
ax6[1].set_xlabel('Tiempo [s]', fontsize=15)
ax6[1].grid()
ax6[2].plot(t_3_rem, senial_fir_3, label='Señal filtrada fir', color='magenta')
ax6[2].legend(loc="upper right", fontsize=15)
ax6[2].set_ylabel('Magnitud [mV])', fontsize=15)
ax6[2].set_xlabel('Tiempo [s]', fontsize=15)
ax6[2].grid()

plt.show()





