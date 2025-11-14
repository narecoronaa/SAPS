# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:24:33 2025

Practica para el recuperatorio del parcial: Examen 29/10/24

Datos del enunciado: 
                ○ Medir la presion de manera no invasiva 
                ○ Sensibilidad de 18 mV/mmHg y un CAD de 16 bits 
                (Vref: 5V) muestreando a 85 Hz
                ○ Parte de este procesamiento consiste en separar, 
                mediante filtrado digital, los pulsos oscilométricos 
                (comprendidos entre 1Hz y20Hz) de la señal de presión de 
                inflado en el manguito (<0.5Hz).

@author: Fernandez Aldana Agustina

"""


#%% include necesarios


#from scipy.signal import find_peaks
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





#%% 1. Determinar la función de transferencia H(s) de un filtro que cumpla 
# con la función de antialiasing y tenga respuesta máximamente plana en la 
# banda de paso. Para ello cuenta con un registro de la señal de presión en 
# el manguito (nibp_1000hz.txt), adquiridos a una frecuencia de muestreo mayor 
# (1kHz), y cuya magnitud está expresada en mV. *


# Para esto planteo que si la frecuencia de muestreo va a ser de 85 Hz, busco a 
# partir de fs/2 el maximo y determino las condiciones que debe cumplir 
# el filtro antialias 

#Grafico tanto la senial como su transformada

archivo ='seniales/nibp_1000hz.txt' # nombre de archivo
senial = np.loadtxt(archivo) # la amplitud de la señal se encuentra en mV
fs = 1000 # frecuencia de muestreo 1000Hz

ts = 1 / fs # tiempo de muestreo
N = len(senial) # número de muestras en el archivo de audio
t = np.linspace(0, N * ts, N) # vector de tiempo -> la función la da x espaciados

# Grafico la senial en el tiempo
fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20))
fig1.suptitle("Senial en el tiempo", fontsize=20)
ax1.set_xlabel('Tiempo [s]', fontsize=15)
ax1.set_ylabel('Magnitud [mV]', fontsize=15)
ax1.grid(True, which="both")
ax1.plot(t, senial, color = 'purple')
plt.show()



frec_senial, fft_senial = fft_mag(senial, fs)

# Grafico la senial en el tiempo
fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20))
fig2.suptitle("Tranformada de la Senial en la frecuencia", fontsize=20)
ax2.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2.set_ylabel('Magnitud [mV]', fontsize=15)
ax2.grid(True, which="both")
ax2.plot(frec_senial, fft_senial, color = 'grey')
plt.show()



FS_resample = 85 
indices_validos = np.where(frec_senial >= FS_resample/2)[0]
magnitudes_recortadas = fft_senial[indices_validos]
frecuencias_recortadas = frec_senial[indices_validos]
frec_max= np.argmax(magnitudes_recortadas)
amp_max = magnitudes_recortadas[frec_max]

indice_fs2 = FS_resample//2
amp_fs_2 = fft_senial[indice_fs2]



#Ahora calculo la atenuacion que necesito cumplir con el filtro antialias
V_REF = 5000                     # Tensión de referencia en V
N_BITS = 16                     # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en V

at_fs2 = 20*np.log10(amp_fs_2/RES)

at_max =  20*np.log10(amp_max/RES) 


print(f"Resolucion del conversor {RES} \n")

print("Los requisitos para el filtro antialisa son: ")

print(f"Atenuación maxima en senial 4: {at_max:.2f}dB en {frec_max:.2f}Hz")
print(f"Atenuación fs/2 a {at_fs2:.2f}dB en {FS_resample/2}Hz \n")


#Implementando un cheby con riplede 0.01 dB, se puede considerar maximamente 
# plano para implementar un filtro de orden 4 de dos estaciones, ya que con 
# el butter es un orden 5 con 3 estaciones.

#%% 2. Implemetar con salley key, poner los valores comerciales para la 
# simulacion y graficarla. 



frec_dis, mag_dis = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')

frec_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')


fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
ax3.set_title('Filtro orden 4', fontsize=18)
ax3.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3.set_ylabel('|H(jw)|² [dB]', fontsize=15)

ax3.set_xscale('log')
ax3.grid(True, which="both")
ax3.plot(frec_dis,  mag_dis, label='Diseñado')
ax3.plot(frec_sim,  mag_sim, label='Simulado')
ax3.plot(FS_resample/2, -at_fs2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(frec_max, -at_max, marker='X', markersize=12, label='Requisito en máximo señal 4')
ax3.legend(loc="lower left", fontsize=15)
plt.show()


#%% 3. Se diseña un filtro iir pasa banda, con la primer banda de rechazo de 0.1 a 1 Hz, 
# banda de paso de 1 a 20Hz, y segunda banda de rechazo de 20 a 30Hz
# con atenuacion en banda de paso de 0,5 dB y en la de rechazo de 62 dB

filtro_iir = np.load('filtros_digitales/filtro_iir.npz', allow_pickle=True) 
print("Filtro IIR: butterworth")
filter_parameters.filter_parameters('filtros_digitales/filtro_iir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_iir, Den_iir = filtro_iir['ba'] 


#%% 4. Se diseña un filtro fir pasa bajo con la frecuencia de corte de banda 
# en 0.5 y de rechazo en 1Hz con una atenuacion de 0.5dB en la banda de 
# paso y 35dB en la de rechazo

filtro_fir = np.load('filtros_digitales/filtro_fir.npz', allow_pickle=True) 
print("Filtro FIR: kaiser")
filter_parameters.filter_parameters('filtros_digitales/filtro_fir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_fir, Den_fir = filtro_fir['ba'] 


#%% 5. Probar el funcionamiento de ambos filtros digitales utilizando la señal de prueba muestreada
# a 85 Hz (nibp_85hz.txt), también expresada en mV. Graficar las señales y sus espectros antes
# y después de ser filtradas.


#Grafico tanto la senial como su transformada

archivo_85 ='seniales/nibp_85hz.txt' # nombre de archivo
senial_resample = np.loadtxt(archivo_85) # la amplitud de la señal se encuentra en mV

ts_resample = 1 / fs # tiempo de muestreo
N_resample = len(senial_resample) # número de muestras en el archivo de audio
t_resample = np.linspace(0, N_resample * ts_resample, N_resample) # vector de tiempo -> la función la da x espaciados

#Aplico el filtro IIR

senial_iir = signal.lfilter(Num_iir, Den_iir, senial_resample)

f_resample, senial_resample_fft_mod = fft_mag(senial_resample, FS_resample)
frec_iir, rta_frec_iir = signal.freqz(Num_iir, Den_iir, worN=f_resample, fs=FS_resample)
f1_iir, senial_iir_fft_mod = fft_mag(senial_iir, FS_resample)


#Aplico el filtro FIR
#Agrego ceros aplico el filtro y luego los eliminto para contrarestar el 
# retardo en el tiempo que implica la implementacion del diltro fir, 
# la cantidad de ceros es el orden del filtro /2

ceros_agregar = filtro_fir['N']//2 

print(f"La mitad del Orden del filtro FIR {ceros_agregar} ")

senial_fir = senial_resample
senial_fir = np.pad(senial_resample,(0,ceros_agregar),mode='constant')
senial_fir = signal.lfilter(Num_fir, Den_fir, senial_fir)
senial_fir = senial_fir[ceros_agregar:] 

frec_fir, rta_frec_fir = signal.freqz(Num_fir, Den_fir, worN=f_resample, fs=FS_resample)
f1_fir, senial_fir_fft_mod = fft_mag(senial_fir, FS_resample)

fig4, ax4 = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Senial remuestreada y las filtradas", fontsize=18)
ax4[0].plot(t_resample, senial_resample, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[0].set_xlabel('Tiempo [s]', fontsize=15)
ax4[0].grid()
ax4[1].plot(t_resample, senial_iir, label='Señal filtrada iir', color='green')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[1].set_xlabel('Tiempo [s]', fontsize=15)
ax4[1].grid()
ax4[2].plot(t_resample, senial_fir, label='Señal filtrada fir', color='magenta')
ax4[2].legend(loc="upper right", fontsize=15)
ax4[2].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[2].set_xlabel('Tiempo [s]', fontsize=15)
ax4[2].grid()
plt.show()



#grafica de la tranformada de la señal a 85Hz y de las filtradas

fig4, ax4 = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Tranformada de la señal original y las filtradas", fontsize=18)
ax4[0].plot(f_resample, senial_resample_fft_mod, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[0].grid()
ax4[1].plot(f1_iir, senial_iir_fft_mod, label='Señal filtrada iir', color='green')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[1].grid()
ax4[2].plot(f1_iir, senial_iir_fft_mod, label='Señal filtrada fir', color='magenta')
ax4[2].legend(loc="upper right", fontsize=15)
ax4[2].set_ylabel('Magnitud [mV])', fontsize=15)
ax4[2].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[2].grid()
plt.show()


#%% 6. Calcular analíticamente la Presión Arterial Media (MAP) en mmHg y la Frecuencia Cardíaca
# en LPM (latidos por minuto). La primera se puede estimar a partir del valor de presión en el
# manguito en el momento en que la señal de pulsos oscilométricos alcanza su máximo (ver
# fig. 1). La segunda, a partir de determinar el valor en frecuencia donde se encuentra el
# máximo en el espectro de la señal de pulsos oscilométricos.


#Para el calculo de la MAP es el instante del maximo en el manguito en la senial de 
# pulsos oscilométricos, y en ese punto a partir de la presion en el manguito puedo determinar la MAP
# Que es la senial obtenida con el filtro fir y utlizamos la Sensibilidad que es un dato para pasar a presion.

sensibilidad = 18 #mV/mmHg 

posicion_inicial = 1000

sub_secuencia = senial_iir[posicion_inicial:]

tiempo_max_pulsos_oscilometricos_relativo = np.argmax(sub_secuencia)
tiempo_max_pulsos_oscilometricos_absoluto = tiempo_max_pulsos_oscilometricos_relativo + posicion_inicial

presion_arterial = senial_fir[tiempo_max_pulsos_oscilometricos_absoluto]/sensibilidad # por la sensibilidad

print(f'La presion arterial media (MAP) es: {presion_arterial} mmHg \n')


#Para la FC se busca en el espectro de la senial de los pulsos oscilometricos. 
posicion_inicial = 100

sub_secuencia = senial_iir[posicion_inicial:] 

frec_card = f1_iir[np.argmax(sub_secuencia) + posicion_inicial] 
frec_card_lpm = frec_card / 60  #1 Hz = 60 lpm
print(f'La frecuencia cardiaca es: {frec_card_lpm} lpm \n')

#nose como calcular la fc





