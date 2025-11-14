# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 21:50:45 2025

@author: narec
"""
import numpy as np
# ---------------------------------------------------------------------------
# Concepto: por qué el filtro antialias debe ser PASA-BAJOS (resumen)
# ---------------------------------------------------------------------------
"""
Resumen conceptual (antialiasing = LOW-PASS):

- Objetivo del filtro antialias: evitar que energía en frecuencias >= Nyquist (fs/2)
  se pliegue (alias) dentro de la banda útil después de la conversión A/D.

- Por qué Pasa-Bajos:
  - Todas las frecuencias por encima de Nyquist (una sola región "alta") tienen
    el potencial de plegarse hacia la banda base. Por tanto, el filtro debe atenuar
    todas esas componentes altas de forma global → se usa un filtro PASA-BAJOS.
  - Un PASABANDA o un PASAALTOS no cumplen: un pasabanda dejaría pasar una banda
    centrada y un pasaaltos dejaría pasar exactamente las frecuencias altas que
    queremos atenuar antes del muestreo.

- Conclusión directa:
  - Antialias = filtro PASA-BAJOS con frecuencia de corte situada antes de Nyquist,
    suficiente atenuación en la región > Nyquist según el nivel de interferencias.
"""

# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
La implementación con componentes pasivos de filtros de Butterworth y Chebyshev I, para
un mismo orden, difiere en el número total de componentes y sus valores (resistencias,
bobinas y capacitores); siendo los primeros los que utilizan menor cantidad de componentes.

Res
la afirmación es falsa porque, para igual orden, el número de componentes reactivos es el mismo; 
sólo difieren los valores y la distribución (y en la práctica la Chebyshev puede necesitar componentes
con tolerancias/valores más extremos).
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
La transformada invariante al impulso permite obtener una versión digital de un sistema
analógico, H(z) a partir de H(s), y debido al aliasing solo puede emplearse para la
implementación de filtros pasa bandas o pasa altos.

Res
la afirmación es falsa — la transformada invariante al impulso toma la respuesta al impulso continua 
h_a(t) y la muestrea para obtener h[n] = T·h_a(nT); en el dominio de polos equivale a mapear polos de 
s a z con z = e^{sT}. Al muestrear la respuesta al impulso la señal en frecuencia del sistema continuo 
se replica (plegada) en múltiplos de la frecuencia de muestreo; esas réplicas (aliasing) distorsionan 
la respuesta digital si hay energía analógica por encima de Nyquist. La técnica NO está limitada 
teóricamente solo a pasa‑banda o pasa‑altos; simplemente es inapropiada cuando el prototipo analógico 
tiene energía significativa a frecuencias altas (por ejemplo muchos diseños pasa‑bajos), porque el 
aliasing arruinará la respuesta digital.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
El circuito de la figura corresponde a un filtro de Sallen-Key. En el mismo se puede variar la
ubicación del par de polos complejos conjugados modificando la relación entre R4 y R3

Res
Verdadero. Al variar la relación R4/R3, se ajusta la ganancia K del amplificador, lo que a su vez modifica el 
factor Q del filtro, cambiando así la ubicación de los polos complejos conjugados.
La ubicación de los polos (que define la respuesta en frecuencia) viene determinada por la frecuencia 
natural (ω₀) y el factor de calidad (Q).

La frecuencia natural ω₀ depende solo de R1, R2, C1, C2.
El factor de calidad Q, que determina cuán agudo es el pico en la frecuencia de corte (y si los polos son 
reales o complejos conjugados), depende directamente de la ganancia K.
"""
# ---------------------------------------------------------------------------
"""
Cuando se diseña un filtro pasa alto analógico utilizando la aproximación de Chebyshev, la
frecuencia de corte del mismo es siempre menor que la frecuencia que define el ancho de
banda de ripple.

Res
falsa. La relación entre la frecuencia de corte y la frecuencia de la banda de ripple depende de cuanto 
ripple se especifique en la banda de paso.

Frecuencia de la banda de ripple: En un filtro pasa alto Chebyshev, es la frecuencia más baja a 
partir de la cual la ganancia entra en la banda de paso y oscila. En este punto, la atenuación es igual 
al ripple máximo especificado.
Frecuencia de corte (f_c): Por definición es la frecuencia donde la atenuación es de -3 dB.

"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
En los filtros de Sallen-Key, sin importar la cantidad de etapas, el número de capacitores es
igual al orden del filtro que se implementa.

Res
La topología de Sallen-Key se basa en la construcción de filtros de órdenes superiores mediante la 
cascada de etapas de primer y segundo orden. La relación entre el número de capacitores y el orden es 
directa. El bloque de construcción fundamental de un filtro Sallen-Key es la etapa de segundo orden 
(que implementa un par de polos). Esta etapa siempre utiliza dos capacitores.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
La implementación del algoritmo para filtrado FIR requiere utilizar tres vectores:
a. Vector de Coeficientes: longitud igual al orden del filtro
b. Vector de Muestras: Longitud igual a la cantidad de muestras a procesar
c. Vector de Resultados: Longitud igual al tamaño del vector de muestras más el tamaño
del vector de coeficientes

Res
La afirmación es falsa porque la longitud del vector de coeficientes es Orden + 1, y la longitud del 
vector de resultados de la convolución completa es 
(longitud de muestras) + (longitud de coeficientes) - 1.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
Si se desea adquirir una señal cuyas componentes de interés se encuentran hasta
100 Hz. ¿Es posible realizar la adquisición libre de aliasing utilizando un filtro
pasabajos de Butterworth de 2do Orden con Frecuencia de corte 100 Hz,
implementado con una celda de Sallen-Key, antes del conversor de 8 bits y emplear
una frecuencia de muestreo de 200 Hz?

Res
Falso. Para evitar el aliasing, el filtro analógico debe atenuar significativamente las frecuencias 
iguales o superiores a la frecuencia de Nyquist (100Hz porque se muestrea a 200Hz). Cualquier señal o 
ruido con una frecuencia  superior a 100 Hz pasará a través del filtro con muy poca atenuación y 
generará aliasing.
"""
# ---------------------------------------------------------------------------
"""
Si el par de polos de una sección de orden 2 se acerca al eje imaginario, el Q de la
misma aumenta.

Res
Verdadero. Cuando los polos se acercan al eje imaginario, el amortiguamiento disminuye. Como Q es 
inversamente proporcional a el amortiguamiento un Q alto significa menos amortiguamiento y una 
respuesta en frecuencia con un pico de resonancia más agudo.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
La gráfica de la figura muestra la magnitud de la respuesta en frecuencia de dos
funciones de chebyshev del mismo orden con diferente ancho de banda de ripple. (grafica roja y celeste)

Res
El título indica "Aproximación de Chebyshev 1 dB". Esto establece que el parámetro de diseño para el 
ripple  en la banda de paso es de 1 dB para ambas curvas mostradas.
Análisis Visual del Ripple: La línea punteada horizontal marca el valle del ripple. En el eje Y, 
|H(w)|² ≈ 0.8. Si convertimos esto a decibeles, obtenemos 10 * log₁₀(0.8) ≈ -0.97 dB, lo que 
corresponde a la atenuación de 1 dB especificada en el título. Ambas curvas, la roja y la celeste, 
tocan esta línea en sus valles, confirmando visualmente que ambas tienen la misma magnitud de ripple.
La diferencia entre las gráficas es el tipo de filtro, no la magnitud del ripple.
"""
# ---------------------------------------------------------------------------
"""
Los conversores sigma-delta incluyen un filtro digital pasabajos como parte del bloque
decimador. 

Res
Verdadero. El conversor sigma-delta funciona en dos etapas principales: el modulador y el decimador.
Este ultimo tiene una función doble, filtrar ruido de alta frecuencia que el modulador generó para 
lo que se usa un filtro digital pasabajos y reducir la frecuencia de muestreo. 
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
Dado un filtro determinado, por ejemplo Butterworth pasabajos de 3er orden con
frecuencia de corte en 20 Hz, si se modifica el Q, se pueden modificar algunos de sus
parámetros pero no deja de ser un filtro de Butterworth. 

Res
El Q no es un parámetro de diseño en un filtro, para un orden y tipo de filtro dados,
el valor de Q de cada par de polos está predeterminado y no se puede cambiar. No es un parámetro que 
se pueda "ajustar". Es una consecuencia directa de la aproximación de Butterworth, Chebyshev, etc.
Modificar el Q cambia la naturaleza del filtro, si fuerzamos un cambio en el Q de una de las etapas, 
se mueven los polos de su ubicación original.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
Se puede aumentar el rango dinámico de un conversor, aumentando la frecuencia de
muestreo de la señal, logrando medio bit de mejora por cada vez que se duplica la
frecuencia de muestreo. 

Res
Verdadero. Esta técnica se cocnoce como sobremuestreo, muestrear el doble de rápido esparce el ruido. 
Al filtrar, quitamos la mitad del ruido, ganando 3 dB, lo que equivale a medio bit.
"""
# ---------------------------------------------------------------------------
"""
Los conversores sigma-delta aplican el concepto de sobremuestreo pero incorporan
un modulador sigma-delta para conformación del ruido.

Res
Verdedero.A diferencia del sobremuestreo simple que solo "diluye" el ruido de cuantización, los 
conversores sigma-delta usan un modulador para hacer "noise shaping". Este proceso empuja la mayor 
parte de la energía del ruido fuera de la banda de interés y la concentra 
en las altas frecuencias permitiendo que el filtro digital posterior elimine el ruido de manera mucho 
más efectiva, logrando un aumento de resolución significativamente mayor.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
Una de las diferencias entre los filtros de Sallen-Key y Múltiple Realimentaciones es
el tipo de aproximación que permiten implementar.

Res
Falso. Ambas topologías son estructuras de circuito que pueden implementar cualquier tipo de 
aproximación, la elección de esta es un paso de diseño matemático que define la ubicación de los polos, 
y cualquiera de estas dos topologías puede ser utilizada para realizar físicamente esas ubicaciones.

Diferencias: Sallen-Key es no inversora y más simple (ideal para 
Q bajo), mientras que MFB es inversora y generalmente más estable para filtros con un Q alto.
"""
# ---------------------------------------------------------------------------
"""
En los filtros FIR de ventana rectangular, al aumentar el tamaño de la ventana
aumenta el orden del filtro.

Res
Verdadero. El orden de un filtro FIR (N) está directamente relacionado con su longitud o número de 
coeficientes (L) por la fórmula N = L - 1. Al diseñar con el método de la ventana, la longitud del 
filtro (L) es igual al tamaño de la ventana que se utiliza.
"""
# ---------------------------------------------------------------------------
"""
La respuesta al impulso de un filtro FIR es siempre de duración menor o igual al
orden del mismo.

Res
Falso. La duración de la respuesta al impulso de un filtro FIR es siempre igual a su orden más uno.
Un filtro de orden N tiene una respuesta al impulso con N+1 coeficientes (desde h[0] hasta h[N]). 
Por lo tanto, su duración es N+1, que es siempre mayor que el orden N. 
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
El algoritmo de filtrado por bloques continuos solapados procesa múltiples ventanas
de una señal de la siguiente manera:
○ Realiza la convolución entre un bloque y la respuesta al impulso del filtro
○ Suma los términos transitorio de salida del procesamiento del bloque anterior a
los términos del transitorio de entrada del procesamiento del bloque actual.

Res
Verdadero. Al procesar una señal por bloques, la convolución de cada bloque con la respuesta al impulso 
del filtro produce un transitorio de salida. Este método reconstruye la señal de salida completa 
sumando esta cola del bloque anterior a la porción inicial del resultado del bloque actual, manejando 
así el solapamiento que se crea.
"""
# ---------------------------------------------------------------------------
def espectro_fft(y, sr):
    """
    Calcula el espectro de frecuencia de una señal 'y' muestreada a 'sr' Hz.
    Usa una ventana de Hanning y normaliza para obtener la amplitud pico.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    # Aplica una ventana de Hanning para suavizar los bordes de la señal
    w = np.hanning(n)
    
    # Calcula la Transformada Rápida de Fourier (FFT)
    Y = np.fft.rfft(y * w)
    
    # Normaliza la magnitud para que el valor corresponda a la amplitud pico
    mag = 2.0 * np.abs(Y) / np.sum(w) 
    
    # --- CORRECCIÓN: Evitar error si mag está vacío ---
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0 # La componente DC (frecuencia 0) no se duplica
    
    # Calcula el vector de frecuencias correspondiente
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    return freqs, mag

"""
El circuito de la figura corresponde a un filtro de Sallen-Key. En el mismo se puede
variar la ubicación y número de polos modificando la relación entre R4 y R3.

Res
Falso.  Este es un circuito de segundo orden, definido por sus dos capacitores (C1 y C2) y por lo tanto
siempre tendrá dos polos. No se puede cambiar el orden del filtro cambiando valores de resistencias.
"""
# ---------------------------------------------------------------------------
"""

"""


