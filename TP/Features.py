# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Procesamiento ML (Machine Learning):
Script ejemplificando la extracción de características temporales de las señales,
y su comparación.     

Autor: Albano Peñalva
Fecha: Febrero 2025

"""
import numpy as np
import pandas as pd
import seaborn as sns
import process_data
import matplotlib.pyplot as plt

#%% Lectura del dataset

FS = 60 # Frecuencia de muestre: 60Hz
T = 3    # Tiempo total de cada registro: 3 segundos

folder = 'dataset_voley_E4' # Carpeta donde se almacenan los .csv

#Procesamos los archivos para obtener el dataset junto con el diccionario de clases
x, y, z, classmap = process_data.process_data(FS, T, folder)
ts = 1 / FS                   # tiempo de muestreo
N = FS*T                      # número de muestras en cada regsitro

#%% Graficación

t = np.linspace(0, N * ts, N)   # vector de tiempos
# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig, axes = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5)

# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se grafica la señal en los tres ejes
            axes[gesture_name][0].plot(t, x[capture, 0:N], label="Trial {}".format(trial_num))
            axes[gesture_name][1].plot(t, y[capture, 0:N], label="Trial {}".format(trial_num))
            axes[gesture_name][2].plot(t, z[capture, 0:N], label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

# Se le da formato a los ejes de cada gráfica
    axes[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    axes[gesture_name][0].grid()
    axes[gesture_name][0].legend(fontsize=6, loc='upper right');
    axes[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    axes[gesture_name][0].set_ylim(-6, 6)
    
    axes[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    axes[gesture_name][1].grid()
    axes[gesture_name][1].legend(fontsize=6, loc='upper right');
    axes[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    axes[gesture_name][1].set_ylim(-6, 6)
    
    axes[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    axes[gesture_name][2].grid()
    axes[gesture_name][2].legend(fontsize=6, loc='upper right');
    axes[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)
    axes[gesture_name][2].set_ylim(-6, 6)

plt.tight_layout()
plt.show()

#%% Extracción de parámetros temporales

# Lista de features
features_list = [
    'x_mean', 'y_mean', 'z_mean', 
    'x_max', 'y_max', 'z_max', 
    'x_min', 'y_min', 'z_min',
    'x_max_pos', 'y_max_pos', 'z_max_pos', 
    'x_min_pos', 'y_min_pos', 'z_min_pos', 
    'x_rms', 'y_rms', 'z_rms', 
    'x_0_cross', 'y_0_cross', 'z_0_cross',
    'x_en_above_0', 'y_en_above_0', 'z_en_above_0',
    'x_en_below_0', 'y_en_below_0', 'z_en_below_0',
    'gesture_name']

# Diccionario de features
fd = {v: i for i, v in enumerate(features_list)}

# Cáclulo de features
features = np.zeros([len(x), len(features_list)])                  
for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
    # Cálculo del valor medio
    features[capture][fd.get('x_mean')] = np.mean(x[capture, 0:N])         
    features[capture][fd.get('y_mean')] = np.mean(y[capture, 0:N])         
    features[capture][fd.get('z_mean')] = np.mean(z[capture, 0:N])    
      
    # TODO: Cálculo del máximo
    features[capture][fd.get('x_max')] = np.max(x[capture, 0:N])
    features[capture][fd.get('y_max')] = np.max(y[capture, 0:N])
    features[capture][fd.get('z_max')] = np.max(z[capture, 0:N])
      
    # TODO: Cálculo del mínimo
    features[capture][fd.get('x_min')] = np.min(x[capture, 0:N])
    features[capture][fd.get('y_min')] = np.min(y[capture, 0:N])
    features[capture][fd.get('z_min')] = np.min(z[capture, 0:N])
    
    # TODO: Cálculo de la posición del máximo
    features[capture][fd.get('x_max_pos')] = np.argmax(x[capture, 0:N])  
    features[capture][fd.get('y_max_pos')] = np.argmax(y[capture, 0:N]) 
    features[capture][fd.get('z_max_pos')] = np.argmax(z[capture, 0:N])  
    
    # TODO: Cálculo de la posición del mínimo
    features[capture][fd.get('x_min_pos')] = np.argmin(x[capture, 0:N]) 
    features[capture][fd.get('y_min_pos')] = np.argmin(y[capture, 0:N]) 
    features[capture][fd.get('z_min_pos')] = np.argmin(z[capture, 0:N])   
     
    # TODO: Cálculo del valor RMS
    features[capture][fd.get('x_rms')] = np.sqrt(np.mean(x[capture, 0:N]**2))
    features[capture][fd.get('y_rms')] = np.sqrt(np.mean(y[capture, 0:N]**2))
    features[capture][fd.get('z_rms')] = np.sqrt(np.mean(z[capture, 0:N]**2))
    
    # TODO: Cálculo de los cruces por cero
    def zero_crosses(seg):
        s = np.sign(seg)
        s[s == 0] = 1  # evitar falsos cruces por ceros exactos
        return np.sum(s[:-1] * s[1:] < 0)
    features[capture][fd.get('x_0_cross')] = zero_crosses(x[capture, 0:N])
    features[capture][fd.get('y_0_cross')] = zero_crosses(y[capture, 0:N])
    features[capture][fd.get('z_0_cross')] = zero_crosses(z[capture, 0:N])   
    
    # SEGMENTOS LOCALES para las energías
    x_seg = x[capture, 0:N]
    y_seg = y[capture, 0:N]
    z_seg = z[capture, 0:N]
    
    # TODO: Cálculo de la energía de la señal por encima de cero (suma de cuadrados de muestras > 0)
    features[capture][fd.get('x_en_above_0')] = np.sum((x_seg[x_seg > 0])**2)
    features[capture][fd.get('y_en_above_0')] = np.sum((y_seg[y_seg > 0])**2)
    features[capture][fd.get('z_en_above_0')] = np.sum((z_seg[z_seg > 0])**2)
    
    # TODO: Cálculo de la energía de la señal por debajo de cero (suma de cuadrados de muestras < 0)
    features[capture][fd.get('x_en_below_0')] = np.sum((x_seg[x_seg < 0])**2)
    features[capture][fd.get('y_en_below_0')] = np.sum((y_seg[y_seg < 0])**2)
    features[capture][fd.get('z_en_below_0')] = np.sum((z_seg[z_seg < 0])**2)  
    
    # gesture_name
    features[capture][fd.get('gesture_name')] = z[capture, N]                     
    
#%% Graficación de matrices de correlación

# Creaciónd e un DataFrame de Pandas
df = pd.DataFrame(features, columns=features_list)

# Graficación de matrices de correlación por cada conjunto de features
sns.pairplot(df, vars=['x_mean', 'y_mean', 'z_mean'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_max', 'y_max', 'z_max'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_min', 'y_min', 'z_min'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_max_pos', 'y_max_pos', 'z_max_pos'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_min_pos', 'y_min_pos', 'z_min_pos'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_rms', 'y_rms', 'z_rms'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_0_cross', 'y_0_cross', 'z_0_cross'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_en_above_0', 'y_en_above_0', 'z_en_above_0'], hue='gesture_name', palette="deep")
sns.pairplot(df, vars=['x_en_below_0', 'y_en_below_0', 'z_en_below_0'], hue='gesture_name', palette="deep")
plt.plot()