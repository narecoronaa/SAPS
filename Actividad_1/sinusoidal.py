#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:46:03 2020

@author: jm
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from   scipy import signal
#import scipy.fftpack as sc
#from scipy.fftpack import fft

#Sinuoidal de 50 Khz con offset de 2 V.

# Parámetros

N  = 1000 # muestras
fs = 1000 # Hz

a0 = 1 # Volts
p0 = 0 # radianes
f0 = 10   # Hz



# comienzo de la función
# tiempo de muestreo    
ts = 1/fs 
#Genero el espacio para poder tener el espacio temporal que va de 0 a N-1
#Flatten convierte a un array de 1 dimensión.
tt = np.linspace(0, (N-1)*ts, N).flatten()
# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía
signal = np.array([], dtype=np.float64).reshape(N,0)        
#Genero la senoidal con los parametros de entrada.
signal = a0 * np.sin(2 * np.pi * f0 * tt + p0)


plt.figure(figsize=(10,6))
plt.plot(tt, signal)
plt.title('Senoidal Amplitud:{} V Fase:{} radianes Frecuencia {} Hz'.format(a0,p0,f0))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.show()    


