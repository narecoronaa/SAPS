# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Procesamiento ML (Machine Learning):
Script ejemplificando la extracción de características temporales de las señales,
y el uso de estas para el entrenamiento mediante ML de un algoritmo de clasificación.     

Autor: Albano Peñalva
Fecha: Febrero 2025

"""
import numpy as np
from sklearn.metrics import recall_score
from micromlgen import port
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import process_data
import process_code
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

#%% Lectura del dataset

FS = 60 # Frecuencia de muestre: 500Hz
T = 3   # Tiempo total de cada registro: 2 segundos

folder = 'dataset_voley_E4' # Carpeta donde se almacenan los .csv

#Procesamos los archivos para obtener el dataset junto con el diccionario de clases
x, y, z, classmap = process_data.process_data(FS, T, folder)
ts = 1 / FS                   # tiempo de muestreo
N = FS*T                      # número de muestras en cada regsitro


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
                         
                      

#%% Entrenamiento del clasificador

#Separamos el dataset del vector con la información que codifica la clase
X, y = features[:, :-1], features[:, -1]

#Separamos el dataset y el vector de salida en partes iguales para el entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#Creamos un modelo de tipo Random Forrest y le pasamos la partición del dataset y salida
# de entrenamiento
clf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features= 10, random_state=0, bootstrap=False).fit(X_train, y_train)

#Utilizo el modelo entrenado para obtener la predicción de las salidas de testo
y_pred_test = clf.predict(X_test)

#Graficamos la matriz de confusión obtenida a partir de los datos de testeo
cm = confusion_matrix(y_test, y_pred_test, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()

#Utilizo la predicción del punto anterior para comparar con la salida real y obtener
# el score del modelo
print("-------Accuracy-------")
print(accuracy_score(y_test, y_pred_test))

print("-------F1 Score-------")
print(f1_score(y_test, y_pred_test, average=None))

print("-------Precision-------")
print(precision_score(y_test, y_pred_test, average=None))

print("-------Recall-------")
print(recall_score(y_test, y_pred_test, average=None))
#Exporto los archivos del modelo para utilizar en la EDU-CIAA
process_code.process_classifier(port(clf, classname="Classifier", classmap=classmap),'RandomForest')

#Graficamos algunos de los arboles
for i in range(10):    
    plt.figure(figsize=(25,15))
    tree.plot_tree(clf.estimators_[i], filled=True, feature_names = features_list, class_names = classmap)
