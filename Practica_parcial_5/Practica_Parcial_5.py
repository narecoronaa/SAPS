# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 19:04:35 202
5
Practica para el recuperatorio del parcial: Examen 01/11/22

Datos del enunciado: 
                ○ sistema de Mantenimiento Predictivo para utilizar en motores
                eléctricos (Bombas, ventiladores, etc) a partir de la señal de
                un acelerómetro midiendo lasvibraciones del dispositivo.
                
                ○ Banda de interes 100 - 2000Hz y atenuacion minima de 20dB en 50Hz
                ○ Frec muestreo 30k Hz


@author: Fernandez Aldana Agustina

"""

#%% Include necesarios 

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


#%% 1. En base a los requerimientos proponga la función de transferencia de un filtro pasabanda
# de respuesta máximamente plana en la banda de paso adecuado para el
# sistema.

# 2. Factorice la H(s) obtenida en el punto anterior e implemente la parte pasa-bajos de
#manera analógica calculando los componentes del circuito activo que la realice. Utilice
#valores comerciales.

#3. Simule el circuito activo obtenido en el punto anterior en LTSpice utilizando valores
#comerciales de componentes y compare la respuesta en frecuencia obtenida con la
#del sistema original. Indique si la considera adecuada.

#4. Analice el filtro analógico obtenido y las señales de acelerómetro para evaluar si
#podría servir como filtro antialias y justifique su respuesta. Se digitalizará a 6kHz, en
#12bits con un voltaje de referencia de 3.3V



#Por lo tanto voy a buscar los requisitos para un filtro pasa bajo analogico, 
# con frecuencia de corte en 2k Hz. y busco a partir fs_remuestreo el maximo que se debe atenuar

# Cargo la senial

archivo = 'seniales/z_axis_30kHz.txt' # nombre de archivo
senial = np.loadtxt(archivo) # carga de la señal
fs = 30000 # frecuencia de muestreo 3000Hz

ts = 1 / fs # tiempo de muestreo
N = len(senial) # número de muestras en el archivo de audio
t = np.linspace(0, N * ts, N) # vector de tiempo -> la función la da x espaciados

# Grafico la senial en el tiempo
fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20))
fig1.suptitle("Senial en el tiempo", fontsize=20)
ax1.set_xlabel('Tiempo [s]', fontsize=15)
ax1.set_ylabel('Magnitud [V]', fontsize=15)
ax1.grid(True, which="both")
ax1.plot(t, senial, color = 'purple')
plt.show()

frec_senial, fft_senial = fft_mag(senial, fs)

fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20))
fig2.suptitle("Tranformada de la senial en la frecuencia", fontsize=20)
ax2.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2.set_ylabel('Magnitud [V]', fontsize=15)
ax2.grid(True, which="both")
ax2.plot(frec_senial, fft_senial, color = 'blue')
plt.show()


FS_resample = 6000
indices_validos = np.where(frec_senial >= FS_resample/2)[0]
magnitudes_recortadas = fft_senial[indices_validos]
frecuencias_recortadas = frec_senial[indices_validos]

indice_max = np.argmax(magnitudes_recortadas)
frec_max = frecuencias_recortadas[indice_max]

amp_max = np.max(magnitudes_recortadas)

amp_fs_2 = fft_senial[FS_resample//2]


# Ya que obtuve el maximo lo paso a decibeles con la resolucion del conversor
# tanto para el maximo como para fs/2

#Ahora calculo la atenuacion que necesito cumplir con el filtro antialias
V_REF = 3.3                     # Tensión de referencia en V
N_BITS = 12                     # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en V

at_fs2 = 20*np.log10(amp_fs_2/RES)

at_max =  20*np.log10(amp_max/RES) 


print(f"Resolucion del conversor {RES} \n")

print("Los requisitos para el filtro antialisa son: \n")
print(f"Atenuación maxima de la senial: {at_max:.2f}dB en {frec_max:.2f}Hz")
print(f"Atenuación fs/2 a {at_fs2:.2f}dB en {FS_resample/2}Hz \n")


#Grafico el filtro diseñado y el simulado con las condiciones de diseño 
# para ver si cumpleo no

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


#Por la grafica se puede ver que cumple bien con la condicion de la maxima atenuacion.
# Vale aclarar que en fs/2 da por encima de cero ya que la tranformada tiene 
# un valor menor que la resolucion, por lo que al calcular la atenuacion esta 
# da positiva, por lo que no se la tiene en cuenta ya que el conversor no percibe


# 5. Implemente la sección restante de manera digital.

# La parte restante seria un pasa banda, con banda de paso de 100 a 2000 Hz y 
# una tenuacion en 50 Hz de minimo 20dB

# Para ello con el PyFDA implemento un filtro irr que cumpla con estas caracteristicas.

filtro_iir = np.load('filtros_digitales/filtro_iir.npz', allow_pickle=True) 
print("Filtro IIR: butterworth")
filter_parameters.filter_parameters('filtros_digitales/filtro_iir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_iir, Den_iir = filtro_iir['ba'] 


#6. Ensaye el filtro digital diseñado sobre la señal z_axis_6kHz.txt, de manera de verificar
#el correcto funcionamiento del filtro y la atenuación obtenida en 50Hz. Dicha señal ya
#ha sido filtrada con un filtro pasa-bajos y remuestreada.


archivo_rem = 'seniales/z_axis_6kHz.txt' # nombre de archivo
senial_rem = np.loadtxt(archivo_rem) # carga de la señal

ts_rem = 1 / FS_resample # tiempo de muestreo
N_rem = len(senial_rem) # número de muestras en el archivo de audio
t_rem = np.linspace(0, N_rem * ts_rem, N_rem) # vector de tiempo -> la función la da x espaciados


senial_iir = signal.lfilter(Num_iir, Den_iir, senial_rem)

f_resample, senial_resample_fft_mod = fft_mag(senial_rem, FS_resample)
frec_iir, rta_frec_iir = signal.freqz(Num_iir, Den_iir, worN=f_resample, fs=FS_resample)
f1_iir, senial_iir_fft_mod = fft_mag(senial_iir, FS_resample)

#Grafico la senial remuestreada y luego de aplicarle el filtro tanto en tiempo como en frecuencia.
fig4, ax4 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Senial remuestreada y las filtradas", fontsize=18)
ax4[0].plot(t_rem, senial_rem, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [V])', fontsize=15)
ax4[0].set_xlabel('Tiempo [s]', fontsize=15)
ax4[0].grid()
ax4[1].plot(t_rem, senial_iir, label='Señal filtrada iir', color='magenta')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [V])', fontsize=15)
ax4[1].set_xlabel('Tiempo [s]', fontsize=15)
ax4[1].grid()

plt.show()



#grafica de la tranformada de la señal a 85Hz y de las filtradas

fig4, ax4 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Tranformada de la señal original y las filtradas", fontsize=18)
ax4[0].plot(f_resample, senial_resample_fft_mod, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [V])', fontsize=15)
ax4[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[0].grid()
ax4[1].plot(f1_iir, senial_iir_fft_mod, label='Señal filtrada iir', color='magenta')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [V])', fontsize=15)
ax4[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[1].grid()
plt.show()


#7. Diseñe un filtro FIR con PyFDA de similares características al obtenido a partir de la
#factorización.
#8. Grafique dos figuras presentando la versión filtrada y la versión original para ambos
# filtros, tanto para el dominio temporal como el frecuencial.

filtro_fir = np.load('filtros_digitales/filtro_fir.npz', allow_pickle=True) 
print("Filtro FIR: kaiser")
filter_parameters.filter_parameters('filtros_digitales/filtro_fir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_fir, Den_fir = filtro_fir['ba'] 

ceros_agregar = filtro_fir['N']//2 

print(f"La mitad del Orden del filtro FIR {ceros_agregar} ")

senial_fir = senial_rem
senial_fir = np.pad(senial_rem,(0,ceros_agregar),mode='constant')
senial_fir = signal.lfilter(Num_fir, Den_fir, senial_fir)
senial_fir = senial_fir[ceros_agregar:] 

frec_fir, rta_frec_fir = signal.freqz(Num_fir, Den_fir, worN=f_resample, fs=FS_resample)
f1_fir, senial_fir_fft_mod = fft_mag(senial_fir, FS_resample)

#Grafico la senial remuestreada y luego de aplicarle el filtro tanto en tiempo como en frecuencia.
fig4, ax4 = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Senial remuestreada y las filtradas", fontsize=18)
ax4[0].plot(t_rem, senial_rem, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [V])', fontsize=15)
ax4[0].set_xlabel('Tiempo [s]', fontsize=15)
ax4[0].grid()
ax4[1].plot(t_rem, senial_fir, label='Señal filtrada fir', color='orange')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [V])', fontsize=15)
ax4[1].set_xlabel('Tiempo [s]', fontsize=15)
ax4[1].grid()
ax4[2].plot(t_rem, senial_iir, label='Señal filtrada iir', color='magenta')
ax4[2].legend(loc="upper right", fontsize=15)
ax4[2].set_ylabel('Magnitud [V])', fontsize=15)
ax4[2].set_xlabel('Tiempo [s]', fontsize=15)
ax4[2].grid()

plt.show()



#grafica de la tranformada de la señal a 85Hz y de las filtradas

fig4, ax4 = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Tranformada de la señal original y las filtradas", fontsize=18)
ax4[0].plot(f_resample, senial_resample_fft_mod, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud [V])', fontsize=15)
ax4[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[0].grid()
ax4[1].plot(f1_fir, senial_fir_fft_mod, label='Señal filtrada fir', color='orange')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud [V])', fontsize=15)
ax4[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[1].grid()
ax4[2].plot(f1_iir, senial_iir_fft_mod, label='Señal filtrada iir', color='magenta')
ax4[2].legend(loc="upper right", fontsize=15)
ax4[2].set_ylabel('Magnitud [V])', fontsize=15)
ax4[2].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[2].grid()
plt.show()



