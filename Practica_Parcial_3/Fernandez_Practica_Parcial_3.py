# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:24:33 2025

Practica para el recuperatorio del parcial: Examen 16-06-22

Datos del enunciado: 
    ● Atenuar la máxima componente en la señal de ruido por debajo de 1.6uV
    ● Banda de interés: 10-200 Hz
    ● Se digitalizará en un conversor AD de 16bits y Vref= 3.3V

Cuenta además con registros de las señales de interés, realizados a una frecuencia de
muestreo de 44.1kHz para el análisis de las mismas:
    ● Señal de audio limpia del corazón: heart.wav
    ● Señal de audio del ruido de interferencia: aa.wav


@author: Fernandez Aldana Agustina
"""

#%% Include necesar
from scipy import signal
#from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from funciones_fft import fft_mag
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
import filter_parameters
from scipy.io import wavfile
#from matplotlib import patches
from scipy.interpolate import interp1d




#%% 1. En base a los requerimientos y la información que pueda obtener de las señales de
#prueba*, proponga una función de transferencia de un filtro pasabanda adecuado
#para el sistema.

ruta_corazon = 'Audios/heart.wav'          # nombre de archivo



fs_cora, audio_cora = wavfile.read( ruta_corazon ) # frecuencia de muestreo y datos de la señal



# Definición de parámetro temporales de la senial corazon 
ts_cora = 1 / fs_cora # tiempo de muestreo
N_cora = len(audio_cora) # número de muestras en el archivo de audio
t_cora = np.linspace(0, N_cora * ts_cora, N_cora) # vector de tiempo -> la función la da x espaciados
senial_cora_0 = audio_cora[:, 0] # se extrae un canal de la pista de audio (si el audio es
                        # estereo-> los estereos tienen 2 canales)
senial_cora_0 = senial_cora_0 * 3.3 / (2 ** 16 - 1) #se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)


senial_cora_1 = audio_cora[:, 1] # se extrae un canal de la pista de audio (si el audio es
                        # estereo-> los estereos tienen 2 canales)
senial_cora_1 = senial_cora_1 * 3.3 / (2 ** 16 - 1) #se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)



ruta_ruido = 'Audios/aa.wav'
fs_ruido, audio_ruido = wavfile.read( ruta_ruido )

ts_ruido = 1 / fs_ruido# tiempo de muestreo
N_ruido = len(audio_ruido) # número de muestras en el archivo de audio
t_ruido = np.linspace(0, N_ruido * ts_ruido, N_ruido) # vector de tiempo -> la función la da x espaciados

senial_ruido_0 = audio_ruido[:, 0] # se extrae un canal de la pista de audio (si el audio es
                        # estereo-> los estereos tienen 2 canales)
senial_ruido_0 = senial_ruido_0 * 3.3 / (2 ** 16 - 1) #se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)

senial_ruido_1 = audio_ruido[:, 1] # se extrae un canal de la pista de audio (si el audio es
                        # estereo-> los estereos tienen 2 canales)
senial_ruido_1 = senial_ruido_1 * 3.3 / (2 ** 16 - 1) #se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)


# Grafico la senial del corazon y el ruido en el tiempo
fig1, ax1 = plt.subplots(2, 1, figsize=(20, 20))
fig1.suptitle("Seniales corazon en el tiempo", fontsize=20)

ax1[0].set_title('Senial de corazon en el tiempo canal 1', fontsize=15)
ax1[0].set_xlabel('Tiempo [s]', fontsize=15)
ax1[0].set_ylabel('Magnitud [V]', fontsize=15)
ax1[0].grid(True, which="both")
ax1[0].plot(t_cora, senial_cora_0)

ax1[1].set_title('Senial de corazon en el tiempo canal 2', fontsize=15)
ax1[1].set_xlabel('Tiempo [s]', fontsize=15)
ax1[1].set_ylabel('Magnitud [V]', fontsize=15)
ax1[1].grid(True, which="both")
ax1[1].plot(t_cora, senial_cora_1)
plt.show()

# Grafico la senial del corazon y el ruido en el tiempo
fig2, ax2 = plt.subplots(2, 1, figsize=(20, 20))
fig2.suptitle("Seniales ruido en el tiempo", fontsize=20)

ax2[0].set_title('Senial de ruido en el tiempo canal 1', fontsize=15)
ax2[0].set_xlabel('Tiempo [s]', fontsize=15)
ax2[0].set_ylabel('Magnitud [V]', fontsize=15)
ax2[0].grid(True, which="both")
ax2[0].plot(t_cora, senial_ruido_0)

ax2[1].set_title('Senial de ruido en el tiempo canal 2', fontsize=15)
ax2[1].set_xlabel('Tiempo [s]', fontsize=15)
ax2[1].set_ylabel('Magnitud [V]', fontsize=15)
ax2[1].grid(True, which="both")
ax2[1].plot(t_ruido, senial_ruido_1)
plt.show()


#%% 2. Se buscan las condiciones que debe cumplir el filtro en cuanto a las 
# atenuacion, esto re realiza analizando la tranformada de la señal de interes 
# y del ruido, (ambas en los dos canales , sonido estereo)

frec_cora_0, fft_senial_cora_0 = fft_mag(senial_cora_0, fs_cora)
frec_cora_1, fft_senial_cora_1 = fft_mag(senial_cora_1, fs_cora)
frec_ruido_0, fft_senial_ruido_0 = fft_mag(senial_ruido_0, fs_ruido)
frec_ruido_1, fft_senial_ruido_1 = fft_mag(senial_ruido_1, fs_ruido)

#Grafico la tranformada de las señales
fig2, ax2 = plt.subplots(4, 1, figsize=(20, 20))
fig2.suptitle("Tranformada de las seniales en la frecuencia", fontsize=18)

ax2[0].set_title('Senial de Corazon canal 1 en la frecuencia', fontsize=15)
ax2[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[0].set_ylabel('Magnitud [V]', fontsize=15)
ax2[0].grid(True, which="both")
ax2[0].plot(frec_cora_0, fft_senial_cora_0)


ax2[1].set_title('Senial de Corazon canal 2 en la frecuencia', fontsize=15)
ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[1].set_ylabel('Magnitud [V]', fontsize=15)
ax2[1].grid(True, which="both")
ax2[1].plot(frec_cora_1, fft_senial_cora_1)


ax2[2].set_title('Senial de Ruido canal 1 en la frecuencia', fontsize=15)
ax2[2].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[2].set_ylabel('Magnitud [V]', fontsize=15)
ax2[2].grid(True, which="both")
ax2[2].plot(frec_ruido_0, fft_senial_ruido_0)

ax2[3].set_title('Senial de Ruido canal 2 en la frecuencia', fontsize=15)
ax2[3].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[3].set_ylabel('Magnitud [V]', fontsize=15)
ax2[3].grid(True, which="both")
ax2[3].plot(frec_ruido_1, fft_senial_ruido_1)
plt.show()


#Busco los maximos a partir de la que despues va a ser la
# frecuencia de muestreo / 750 Hz
fb = 200
FS_resample = 1500 
indices_validos_1 = np.where(frec_cora_0 >= FS_resample/2)[0]
indices_validos_2 = np.where(frec_cora_1 >= FS_resample/2)[0]
indices_validos_3 = np.where(frec_ruido_0 >= FS_resample/2)[0]
indices_validos_4 = np.where(frec_ruido_1 >= FS_resample/2)[0]

#Recorto el vector de busqueda, las magnitudes y frecuencias correspondientes
magnitudes_recortadas_1 = fft_senial_cora_0[indices_validos_1]
magnitudes_recortadas_2 = fft_senial_cora_1[indices_validos_2]
magnitudes_recortadas_3 = fft_senial_ruido_0[indices_validos_3]
magnitudes_recortadas_4 = fft_senial_ruido_1[indices_validos_4]


frecuencias_recortadas_1 = frec_cora_0[indices_validos_1]
frecuencias_recortadas_2 = frec_cora_1[indices_validos_2]
frecuencias_recortadas_3 = frec_ruido_0[indices_validos_3]
frecuencias_recortadas_4 = frec_ruido_1[indices_validos_4]

# Buscar el índice del máximo dentro de cada subconjunto
indice_max_1 = np.argmax(magnitudes_recortadas_1)
indice_max_2 = np.argmax(magnitudes_recortadas_2)
indice_max_3 = np.argmax(magnitudes_recortadas_3)
indice_max_4 = np.argmax(magnitudes_recortadas_4)

# Con el indice busco a que amplitud corresponde en cada seniañ
amp_max_1 = magnitudes_recortadas_1[indice_max_1]
amp_max_2 = magnitudes_recortadas_2[indice_max_2]
amp_max_3 = magnitudes_recortadas_3[indice_max_3]
amp_max_4 = magnitudes_recortadas_4[indice_max_4]

indice_fs2 = FS_resample//2
amp_1_fs_2 = fft_senial_cora_0[indice_fs2]
amp_2_fs_2 = fft_senial_cora_1[indice_fs2]
amp_3_fs_2 = fft_senial_ruido_0[indice_fs2]
amp_4_fs_2 = fft_senial_ruido_1[indice_fs2]

frecuencia_max_1 = frecuencias_recortadas_1[indice_max_1]
frecuencia_max_2 = frecuencias_recortadas_2[indice_max_2]
frecuencia_max_3 = frecuencias_recortadas_3[indice_max_3]
frecuencia_max_4 = frecuencias_recortadas_4[indice_max_4]

print("En la senial de voz 1")
print(f"Máximo valor: {amp_max_1} a {frecuencia_max_1} Hz")
print(f"Valor en fs/2: {amp_1_fs_2} a {FS_resample/2} Hz \n")

print("En la senial de voz 2")
print(f"Máximo valor: {amp_max_2} a {frecuencia_max_2} Hz")
print(f"Valor en fs/2: {amp_2_fs_2} a {FS_resample/2} Hz \n")

print("En la senial de voz 3")
print(f"Máximo valor: {amp_max_3} a {frecuencia_max_3} Hz")
print(f"Valor en fs/2: {amp_3_fs_2} a {FS_resample/2} Hz \n")

print("En la senial de voz 4")
print(f"Máximo valor: {amp_max_4} a {frecuencia_max_4} Hz")
print(f"Valor en fs/2: {amp_4_fs_2} a {FS_resample/2} Hz \n")

#Busco el maximo de las fs/2
max_fs2 = 0
if(max_fs2 < amp_1_fs_2):
    max_fs2 = amp_1_fs_2

if(max_fs2 < amp_2_fs_2):
    max_fs2 = amp_2_fs_2

if(max_fs2 < amp_3_fs_2):
    max_fs2 = amp_3_fs_2
    
if(max_fs2 < amp_4_fs_2):
    max_fs2 = amp_4_fs_2

#Ahora calculo la atenuacion que necesito cumplir con el filtro antialias
V_REF = 3.3                     # Tensión de referencia en V
N_BITS = 16                     # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en V

at_fs2_2 = 20*np.log10(max_fs2/RES)

at_1 =  20*np.log10(amp_max_1/RES) 
at_2 =  20*np.log10(amp_max_2/RES) 
at_3 =  20*np.log10(amp_max_3/RES) 
at_4 =  20*np.log10(amp_max_4/RES) 



print(f"Banda de paso hasta {fb} Hz")   
print(f"Resolucion del conversor {RES} \n")


#Muestro todas las atenuaciones.

print("Los requisitos para el filtro antialisa son: ")
print(f"Atenuación maxima en senial 1: {at_1:.2f}dB en {frecuencia_max_1:.3f}Hz")
print(f"Atenuación maxima en senial 2: {at_2:.2f}dB en {frecuencia_max_2:.3f}Hz")
print(f"Atenuación maxima en senial 3: {at_3:.2f}dB en {frecuencia_max_3:.3f}Hz")
print(f"Atenuación maxima en senial 4: {at_4:.2f}dB en {frecuencia_max_4:.3f}Hz")
print(f"Atenuación fs/2 a {at_fs2_2:.2f}dB en {FS_resample/2}Hz \n")



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
ax3.plot(FS_resample/2, -at_fs2_2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(frecuencia_max_1, -at_1, marker='X', markersize=12, label='Requisito en máximo señal 1')
ax3.plot(frecuencia_max_2, -at_2, marker='X', markersize=12, label='Requisito en máximo señal 2')
ax3.plot(frecuencia_max_3, -at_3, marker='X', markersize=12, label='Requisito en máximo señal 3')
ax3.plot(frecuencia_max_4, -at_4, marker='X', markersize=12, label='Requisito en máximo señal 4')
ax3.legend(loc="lower left", fontsize=15)
plt.show()

#%% 3. Para saber si se puede implementar como filtro anti alias debe cumplir con 
#atenuar las peores condiciones a partir de la nueva frecuencia de muestreo /2 
#(750 Hz) por lo que comparo esta atenuacion que debe cumplir en base al ruido 
#dado, por otro lado si el filtro planteado cumle con eso.


# Crear función de interpolación
interp_db = interp1d(frec_sim, mag_sim, kind='linear', fill_value="extrapolate")

# Consultar atenuación a 750 Hz
f_objetivo = 750
sim_frec_2 = -1*interp_db(f_objetivo)


print(f"Atenuación del filtro simulado en {FS_resample/2} Hz es {sim_frec_2} dB debe ser mayor que {at_fs2_2 }dB \n")

#Atenuación del filtro simulado en 750.0 Hz es 46.20073084896904 dB debe ser mayor que 45.35852728425671 dB
#El filtro cumple!


#%%  4. Utilizando la herramienta pyFDA diseñe un filtro de tipo IIR de 
# Chebyshev y otro tipo FIR Kaiser, implemento en la nueva senial proporcionada
# que esta muestreada a 1500 Hz


#Determino los indices validos (los que corresponden a frecuencias mayores a FS/2) 

ruta_cora_1500 = 'Audios/heart_1500.wav'
fs_cora_1500, audio_cora_1500 = wavfile.read( ruta_cora_1500 )

ts_cora_1500 = 1 / fs_cora_1500# tiempo de muestreo
N_cora_1500 = len(audio_cora_1500) # número de muestras en el archivo de audio
t_cora_1500 = np.linspace(0, N_cora_1500 * ts_cora_1500, N_cora_1500) # vector de tiempo -> la función la da x espaciados


senial_cora_1500 = audio_cora_1500 * 3.3 / (2 ** 16 - 1) #se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)



#Filtrado IIR

filtro_iir = np.load('filtros_digitales/filtro_iir.npz', allow_pickle=True) 
print("Filtro IIR: butterworth")
filter_parameters.filter_parameters('filtros_digitales/filtro_iir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
Num_iir, Den_iir = filtro_iir['ba'] 
senial_iir = signal.lfilter(Num_iir, Den_iir, senial_cora_1500)

f, senial_fft_mod = fft_mag(senial_cora_1500, FS_resample)
frec_iir, rta_frec_iir = signal.freqz(Num_iir, Den_iir, worN=f, fs=FS_resample)
f1_iir, senial_iir_fft_mod = fft_mag(senial_iir, FS_resample)




#Filtrado FIR

filtro_fir = np.load('filtros_digitales/filtro_fir.npz', allow_pickle=True) 
print("\r\n")
print("Filtro FIR: Kaiser")
filter_parameters.filter_parameters('filtros_digitales/filtro_fir.npz')
print("\r\n")


# Se extraen los coeficientes de numerador y denominador
Num_fir, Den_fir = filtro_fir['ba'] 

#Agrego ceros aplico el filtro y luego los eliminto para contrarestar el 
# retardo en el tiempo que implica la implementacion del diltro fir, 
# la cantidad de ceros es el orden del filtro /2

ceros_agregar = filtro_fir['N']//2 

print(f"La mitad del Orden del filtro FIR {ceros_agregar} ")

senial_fir = senial_cora_1500
senial_fir = np.pad(senial_cora_1500,(0,ceros_agregar),mode='constant')
senial_fir = signal.lfilter(Num_fir, Den_fir, senial_fir)
senial_fir = senial_fir[ceros_agregar:] 

frec_fir, rta_frec_fir = signal.freqz(Num_fir, Den_fir, worN=f, fs=FS_resample)
f1_fir, senial_fir_fft_mod = fft_mag(senial_fir, FS_resample)



fig5, ax5 = plt.subplots(1, 1, figsize=(18, 12))
ax5.set_title('Senial con frecuencia de muestreo 1500 Hz', fontsize=18)
ax5.plot(t_cora_1500, senial_cora_1500, label='Senial Original' )
ax5.legend(loc="upper right", fontsize=15)
ax5.set_xlabel('Tiempo [s]', fontsize=15)
ax5.set_ylabel('Magnitud [V]', fontsize=15)
ax5.grid(True, which="both")
ax5.plot(t_cora_1500, senial_iir, label='Senial Filtrada IIR', color = 'green')
ax5.legend(loc="upper right", fontsize=15)
ax5.plot(t_cora_1500, senial_fir, label='Senial Filtrada FIR', color ='magenta')
ax5.legend(loc="upper right", fontsize=15)
ax5.grid()
plt.show()


#grafica de la tranformada de la señal a 1500Hz y de las filtradas

fig4, ax4 = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig4.suptitle("Tranformada de la señal original y la filtrada", fontsize=18)
ax4[0].plot(f, senial_fft_mod, label='Señal Original', color='blue')
ax4[0].legend(loc="upper right", fontsize=15)
ax4[0].set_ylabel('Magnitud )', fontsize=15)
ax4[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[0].grid()
ax4[1].plot(f1_iir, senial_iir_fft_mod, label='Señal filtrada iir', color='green')
ax4[1].legend(loc="upper right", fontsize=15)
ax4[1].set_ylabel('Magnitud )', fontsize=15)
ax4[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[1].grid()
ax4[2].plot(f1_iir, senial_iir_fft_mod, label='Señal filtrada fir', color='magenta')
ax4[2].legend(loc="upper right", fontsize=15)
ax4[2].set_ylabel('Magnitud )', fontsize=15)
ax4[2].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax4[2].grid()
plt.show()




fig6, ax6 = plt.subplots(1, 1, figsize=(15, 15), sharex=True)
fig6.suptitle("Tranformada de los filtros ", fontsize=18)
ax6.plot(frec_fir, rta_frec_fir, label='Filtro fir', color='magenta')
ax6.legend(loc="upper right", fontsize=15)
ax6.plot(frec_iir, rta_frec_iir, label='Filtro iir', color='green')
ax6.legend(loc="upper right", fontsize=15)
ax6.set_ylabel('Magnitud )', fontsize=15)
ax6.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax6.grid()
plt.show()
