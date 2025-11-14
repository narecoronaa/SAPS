# -*- coding: utf-8 -*-
"""
Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Ejercicios 3 y 4: Filtro IIR (pasa banda) y Envolvente FIR (pasa bajos)
VERSIÓN PASO A PASO CON FUNCIONES SEPARADAS Y COMENTARIOS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import sys 

# --------------------------------------------------------------------------
# FUNCIÓN 1: CÁLCULO DE ESPECTRO (FFT)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# FUNCIÓN 2: CARGAR LA SEÑAL DESDE EL CSV
# --------------------------------------------------------------------------
def cargar_senal(archivo_csv, canal_nombre):
    """
    Carga una señal desde un archivo CSV usando pandas.
    Devuelve solo el canal (columna) especificado.
    Además, si el CSV contiene las columnas 'Ax','Ay','Az', las guarda en
    variables globales Ax_500hz, Ay_500hz, Az_500hz para su uso posterior.
    """
    print(f"  Paso 2a: Intentando leer '{archivo_csv}'...")
    try:
        # Lee el archivo CSV. Asume que la primera fila es la cabecera.
        df = pd.read_csv(archivo_csv)
        
        # Intenta extraer la columna (canal) por su nombre
        senial_original = df[canal_nombre].values
        
        # Si existen las tres componentes, guardarlas en variables globales
        if set(['Ax', 'Ay', 'Az']).issubset(df.columns):
            global Ax_500hz, Ay_500hz, Az_500hz
            Ax_500hz = df['Ax'].values
            Ay_500hz = df['Ay'].values
            Az_500hz = df['Az'].values
            print(f"  Paso 2b: Columnas 'Ax','Ay','Az' extraídas y guardadas en variables globales.")
        else:
            print(f"  Nota: No se encontraron las tres columnas 'Ax','Ay','Az' juntas. Se devolverá sólo '{canal_nombre}'.")

        print(f"  Paso 2c: Columna '{canal_nombre}' extraída exitosamente.")
        
        # --- ¡CORRECCIÓN DE SYNTAXERROR! ---
        # Esta línea DEBE estar DENTRO del bloque 'try'
        return senial_original
        
    except FileNotFoundError:
        print(f"  ERROR: ¡No se encontró el archivo '{archivo_csv}'!")
        print("         Asegúrate de que esté en la misma carpeta que el script.")
        return None # Devuelve None para que el script se detenga
        
    except KeyError:
        print(f"  ERROR: El archivo CSV no tiene una columna llamada '{canal_nombre}'.")
        print(f"         Columnas encontradas: {df.columns.to_list()}")
        return None # Devuelve None para que el script se detenga

# --------------------------------------------------------------------------
# FUNCIÓN 3: REMUESTREAR LA SEÑAL (Usando signal.resample)
# --------------------------------------------------------------------------
def remuestrear_senal(senal, fs_in, fs_out):
    """
    Remuestrea la señal de 'fs_in' a 'fs_out' usando signal.resample,
    calculando el nuevo número de muestras (método de tu script de ejemplo).
    """
    print(f"  Paso 3: Remuestreando señal de {fs_in} Hz a {fs_out} Hz...")
    
    # 1. Obtener el número original de muestras
    n_original = len(senal)
    
    # 2. Calcular el nuevo número de muestras (N_resample)
    n_resample = int(n_original * (fs_out / fs_in))
    
    # 3. Aplicar el remuestreo usando signal.resample
    senial_resampleada = signal.resample(senal, n_resample)
    
    print(f"  Señal remuestreada de {n_original} a {n_resample} muestras.")
    
    return senial_resampleada

# --------------------------------------------------------------------------
# FUNCIÓN 4: CARGAR LOS COEFICIENTES DEL FILTRO (npz exportado por pyFDA)
# --------------------------------------------------------------------------
def cargar_filtro(archivo_npz):
    """
    Carga los coeficientes 'b' (numerador) y 'a' (denominador)
    desde un archivo .npz guardado por pyFDA.
    Funciona tanto para IIR como para FIR.
    """
    print(f"  Paso 4: Cargando coeficientes desde '{archivo_npz}'...")
    try:
        # Carga el archivo .npz
        filtro_data = np.load(archivo_npz, allow_pickle=True)
        
        # Extrae los coeficientes 'ba' (estándar de pyFDA para IIR/FIR)
        b, a = filtro_data['ba']
        
        print(f"  Coeficientes 'b' (orden {len(b)-1}) y 'a' cargados.")
        return b, a
        
    except FileNotFoundError:
        print(f"  ERROR: ¡No se encontró el archivo del filtro '{archivo_npz}'!")
        return None, None
    except Exception as e:
        print(f"  ERROR cargando filtro: {e}")
        return None, None

# --------------------------------------------------------------------------
# FUNCIÓN 5: APLICAR EL FILTRO A LA SEÑAL (genérico: FIR/IIR)
# --------------------------------------------------------------------------
def aplicar_filtro(b, a, senal, usar_filfilt=True):
    """
    Aplica un filtro definido por coeficientes b,a a 'senal'.
    Usa 'filtfilt' (fase cero) por defecto.
    Usa 'lfilter' (causal) si usar_filfilt=False.
    """
    print(f"  Aplicando filtro (Orden N={len(b)-1})...")
    if b is None or a is None:
        raise ValueError("Coeficientes del filtro inválidos.")
    try:
        if usar_filfilt:
            # filtfilt (doble pasada) para no introducir delay de fase
            return signal.filtfilt(b, a, senal)
        else:
            # lfilter (causal) introduce delay
            return signal.lfilter(b, a, senal)
    except Exception as e:
        print(f"  Warning: filtfilt failed ({e}), usando lfilter como fallback.")
        return signal.lfilter(b, a, senal)

# --------------------------------------------------------------------------
# --- NUEVAS FUNCIONES (EJERCICIO 4) ---
# --------------------------------------------------------------------------
def calcular_magnitud(ax, ay, az):
    """
    Calcula la magnitud vectorial de la aceleración por muestra:
      mag = sqrt(ax^2 + ay^2 + az^2)
    """
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    return mag

def rectificar_senal(senal):
    """
    Rectifica la señal (valor absoluto).
    """
    return np.abs(senal)

def calcular_envolvente(ax, ay, az, b_fir, a_fir):
    """
    Calcula la envolvente completa:
      1) magnitud = sqrt(ax^2+ay^2+az^2)
      2) rectificado = abs(magnitud)
      3) envolvente = filtrado pasa-bajos FIR (b_fir, a_fir)
    """
    print("  Calculando magnitud...")
    mag = calcular_magnitud(ax, ay, az)
    
    print("  Rectificando magnitud...")
    rect = rectificar_senal(mag)
    
    print("  Filtrando magnitud para obtener envolvente...")
    # Usamos filtfilt (fase cero) para que la envolvente se alinee
    # con la señal de magnitud, sin retraso (delay).
    envolvente = aplicar_filtro(b_fir, a_fir, rect, usar_filfilt=True)
    return rect, envolvente

# --------------------------------------------------------------------------
# FUNCIÓN 6: GRAFICAR RESULTADOS (EJERCICIO 3)
# --------------------------------------------------------------------------
def graficar_resultados(t, senial_antes, senial_despues, 
                        freqs_antes, mag_antes, freqs_despues, mag_despues, 
                        fs_nueva, banda_interes, canal, archivo_salida):
    """
    Genera los 4 gráficos (temporal y espectro, antes y después)
    y los guarda en un archivo.
    """
    print(f"  Paso 6: Generando gráficos de Ejercicio 3 para canal {canal}...")
    
    with np.errstate(divide='ignore'):
        mag_antes_db = 20 * np.log10(mag_antes)
        mag_despues_db = 20 * np.log10(mag_despues)
    
    mag_antes_db[np.isneginf(mag_antes_db)] = -120
    mag_despues_db[np.isneginf(mag_despues_db)] = -120
    
    lim_y_espectro = -120
    if (freqs_antes.size > 0 and np.any(mag_antes_db[freqs_antes > 0.5])):
        lim_y_espectro = np.max(mag_antes_db[freqs_antes > 0.5]) + 5
    if (freqs_despues.size > 0 and np.any(mag_despues_db[freqs_despues > 0.5])):
        lim_y_desp = np.max(mag_despues_db[freqs_despues > 0.5]) + 5
        if lim_y_desp > lim_y_espectro:
            lim_y_espectro = lim_y_desp
    if lim_y_espectro <= -100: lim_y_espectro = 0 # Fallback

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Ejercicio 3: Filtrado IIR Pasa-Banda (1-11 Hz) sobre Canal '{canal}'", fontsize=16)

    # Gráfico 1: Temporal ANTES
    axes[0, 0].plot(t, senial_antes, label=f'Señal Original ({fs_nueva} Hz)')
    axes[0, 0].set_title('Señal Temporal (Antes de Filtrar)')
    axes[0, 0].set_xlabel('Tiempo (s)')
    axes[0, 0].set_ylabel('Amplitud [g]')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Gráfico 2: Temporal DESPUÉS
    axes[0, 1].plot(t, senial_despues, label='Señal Filtrada', color='orange')
    axes[0, 1].set_title('Señal Temporal (Después de Filtrar)')
    axes[0, 1].set_xlabel('Tiempo (s)')
    axes[0, 1].set_ylabel('Amplitud [g]')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Gráfico 3: Espectro ANTES
    axes[1, 0].plot(freqs_antes, mag_antes_db, label='Espectro Original')
    axes[1, 0].set_title('Espectro (Antes de Filtrar)')
    axes[1, 0].set_xlabel('Frecuencia (Hz)')
    axes[1, 0].set_ylabel('Magnitud (dB)')
    axes[1, 0].axvspan(banda_interes[0], banda_interes[1], color='g', alpha=0.3, label='Banda de Interés')
    axes[1, 0].axvline(x=50, color='r', linestyle='--', label='Ruido (50 Hz)')
    axes[1, 0].set_xlim(0, fs_nueva / 2) # Mostrar hasta Nyquist (60 Hz)
    axes[1, 0].set_ylim(-100, lim_y_espectro) 
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Gráfico 4: Espectro DESPUÉS
    axes[1, 1].plot(freqs_despues, mag_despues_db, label='Espectro Filtrado', color='orange')
    axes[1, 1].set_title('Espectro (Después de Filtrar)')
    axes[1, 1].set_xlabel('Frecuencia (Hz)')
    axes[1, 1].set_ylabel('Magnitud (dB)')
    axes[1, 1].axvspan(banda_interes[0], banda_interes[1], color='g', alpha=0.3, label='Banda de Interés')
    axes[1, 1].axvline(x=50, color='r', linestyle='--', label='Ruido (50 Hz)')
    axes[1, 1].set_xlim(0, fs_nueva / 2) # Mostrar hasta Nyquist (60 Hz)
    axes[1, 1].set_ylim(-100, lim_y_espectro)
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(archivo_salida)
    print(f"  Gráficos guardados en '{archivo_salida}'")
    plt.show()

# --------------------------------------------------------------------------
# --- SCRIPT PRINCIPAL (Aquí comienza la ejecución "paso a paso") ---
# --------------------------------------------------------------------------
if __name__ == "__main__":

    # --- PASO 0: Definir Parámetros ---
    FS_ORIGINAL = 500      # Fs del CSV original
    FS_NUEVA = 120         # Fs elegida en Ej. 2 (para remuestreo y filtro)
    
    ARCHIVO_CSV = 'dataset\caida1_500Hz.csv' # Asegúrate que está en la misma carpeta
    
    # Nombres de tus filtros diseñados en pyFDA
    ARCHIVO_FILTRO_IIR = 'Filtro_IIR_Chebyshev.npz' # Ej. 3 (Pasa-Banda 1-11 Hz)
    ARCHIVO_FILTRO_FIR = 'Filtro_FIR_Equiripple.npz' # Ej. 4 (Pasa-Bajos 1 Hz)
    
    BANDA_INTERES = [1, 11] # [Hz]

    print(f"--- Ejecutando Ejercicios 3 y 4 ---")
    print(f"Fs Original: {FS_ORIGINAL} Hz | Fs Nueva: {FS_NUEVA} Hz")

    try:
        # --- Carga de Filtros (Se cargan primero) ---
        print("\nPASO A: Cargando filtro IIR (Pasa-Banda)...")
        b_iir, a_iir = cargar_filtro(ARCHIVO_FILTRO_IIR)
        if b_iir is None:
            raise Exception(f"No se pudo cargar el filtro IIR: {ARCHIVO_FILTRO_IIR}")

        print("\nPASO B: Cargando filtro FIR (Envolvente)...")
        b_fir, a_fir = cargar_filtro(ARCHIVO_FILTRO_FIR)
        if b_fir is None:
            raise Exception(f"No se pudo cargar el filtro FIR: {ARCHIVO_FILTRO_FIR}")

        # --- Carga de Señales (se cargan las globales Ax,Ay,Az) ---
        print("\nPASO C: Cargando señales (Ax, Ay, Az)...")
        # Usamos 'Ax' como señal de prueba, pero esto carga las 3 globales
        senial_500hz_ax = cargar_senal(ARCHIVO_CSV, 'Ax') 
        if senial_500hz_ax is None or 'Ax_500hz' not in globals():
            raise Exception(f"No se pudo cargar la señal o las variables globales desde {ARCHIVO_CSV}")
        
        # ==================================================================
        # === INICIO BLOQUE AAGREGADO: ANÁLISIS FILTRO ANTI-ALIASING (AAF) ===
        # ==================================================================
        print("\n" + "="*50)
        print("COMIENZA ANÁLISIS DE REQUISITOS PARA FILTRO ANTI-ALIASING (AAF)")
        print("="*50)

        # 1. Definir parámetros del sistema de adquisición de destino
        N_BITS_ADC = 16
        RANGO_ADC = 16  # Asumimos un acelerómetro de +/-16g
        FULL_SCALE_RANGE = RANGO_ADC * 2
        LSB_G = FULL_SCALE_RANGE / (2**N_BITS_ADC) # Resolución del ADC en 'g'
        F_NYQUIST_NUEVA = FS_NUEVA / 2 # Frecuencia de Nyquist de nuestro sistema final

        print(f"  Parámetros del sistema destino:")
        print(f"  Fs = {FS_NUEVA} Hz -> F_nyquist = {F_NYQUIST_NUEVA} Hz")
        print(f"  Resolución ADC = {N_BITS_ADC} bits con rango de +/-{RANGO_ADC}g")
        print(f"  1 LSB equivale a {LSB_G * 1000:.4f} mili-g")

        # 2. Calcular espectro de las señales originales de 500 Hz
        print("\n  Calculando espectros de las señales originales (500 Hz)...")
        freqs_ax, mag_ax = espectro_fft(Ax_500hz, FS_ORIGINAL)
        freqs_ay, mag_ay = espectro_fft(Ay_500hz, FS_ORIGINAL)
        freqs_az, mag_az = espectro_fft(Az_500hz, FS_ORIGINAL)

        # 3. Encontrar el peor caso (máximo) en la banda de rechazo (f > F_nyquist)
        peor_amplitud = 0
        peor_frecuencia = 0
        peor_eje = ''

        for eje, mag, freqs in [('Ax', mag_ax, freqs_ax), ('Ay', mag_ay, freqs_ay), ('Az', mag_az, freqs_az)]:
            # Buscamos en la región de frecuencias que causaría aliasing
            indices_rechazo = np.where(freqs >= F_NYQUIST_NUEVA)
            if len(indices_rechazo[0]) > 0:
                max_amp_eje = np.max(mag[indices_rechazo])
                if max_amp_eje > peor_amplitud:
                    peor_amplitud = max_amp_eje
                    # Encontrar la frecuencia de ese máximo
                    peor_frecuencia = freqs[indices_rechazo][np.argmax(mag[indices_rechazo])]
                    peor_eje = eje
        
        print(f"\n  Análisis del peor caso para Aliasing:")
        print(f"  Componente máxima encontrada en banda de rechazo (f >= {F_NYQUIST_NUEVA} Hz):")
        print(f"    - Eje: {peor_eje}")
        print(f"    - Frecuencia: {peor_frecuencia:.2f} Hz")
        print(f"    - Amplitud: {peor_amplitud:.4f} [g]")

        # 4. Calcular la atenuación requerida
        if peor_amplitud > 0:
            # La atenuación debe llevar la 'peor_amplitud' por debajo del valor de 1 LSB
            atenuacion_requerida_db = 20 * np.log10(peor_amplitud / LSB_G)
            print(f"\n  Atenuación MÍNIMA requerida para el AAF a {peor_frecuencia:.2f} Hz: {atenuacion_requerida_db:.2f} dB")
        else:
            atenuacion_requerida_db = 0
            print("\n  No se encontraron componentes significativas en la banda de rechazo. No se requiere atenuación.")

        # 5. Graficar el resultado para visualizar el requisito
        plt.figure(figsize=(12, 7))
        plt.suptitle("Análisis de Requisitos del Filtro Anti-Aliasing (AAF)", fontsize=16)

        # Usamos el espectro del peor eje encontrado
        if peor_eje == 'Ax': freqs_plot, mag_plot = freqs_ax, mag_ax
        elif peor_eje == 'Ay': freqs_plot, mag_plot = freqs_ay, mag_ay
        else: freqs_plot, mag_plot = freqs_az, mag_az

        mag_db = 20 * np.log10(mag_plot)
        mag_db[np.isneginf(mag_db)] = -140 # Limpiar infinitos

        plt.plot(freqs_plot, mag_db, label=f'Espectro original del eje {peor_eje} (500 Hz)')
        plt.axvline(x=F_NYQUIST_NUEVA, color='red', linestyle='--', label=f'Frecuencia Nyquist Destino ({F_NYQUIST_NUEVA} Hz)')
        
        # Marcar el punto del peor caso
        plt.plot(peor_frecuencia, 20*np.log10(peor_amplitud), 'x', color='black', markersize=12, mew=3,
                 label=f'Peor caso: {peor_amplitud:.2f}g @ {peor_frecuencia:.1f}Hz')
        
        # Dibujar el "piso de ruido" del ADC
        lsb_db = 20*np.log10(LSB_G)
        plt.axhline(y=lsb_db, color='green', linestyle=':', label=f'Nivel de 1 LSB ({lsb_db:.1f} dBg)')
        
        # Anotar la atenuación requerida
        plt.annotate(
            f'Atenuación Req.\n{atenuacion_requerida_db:.1f} dB',
            xy=(peor_frecuencia, 20*np.log10(peor_amplitud)),
            xytext=(peor_frecuencia + 10, lsb_db + 20),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center'
        )

        plt.title(f'Espectro de la señal original y requisito del AAF')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Magnitud [dBg]')
        plt.grid(True, which='both')
        plt.legend()
        plt.xlim(0, FS_ORIGINAL/2)
        plt.ylim(-120, max(mag_db) + 10)
        plt.savefig("analisis_aaf_requisitos.png")
        plt.show()

        # ==================================================================
        # === FIN BLOQUE AGREGADO ========================================
        # ==================================================================
        
        
        # --- EJERCICIO 3: Procesar un canal ('Ax') ---
        print("\n" + "="*50)
        print("COMIENZA EJERCICIO 3: FILTRADO IIR (Canal Ax)")
        print("="*50)
        
        # 3.1: Remuestrear 'Ax'
        senial_120hz_antes = remuestrear_senal(senial_500hz_ax, FS_ORIGINAL, FS_NUEVA)
        t_120hz = np.arange(len(senial_120hz_antes)) / FS_NUEVA
        
        # 3.2: Aplicar filtro IIR
        senial_120hz_despues = aplicar_filtro(b_iir, a_iir, senial_120hz_antes, usar_filfilt=False) # Usar lfilter causal
        
        # 3.3: Calcular espectros y graficar
        freqs_antes, mag_antes = espectro_fft(senial_120hz_antes, FS_NUEVA)
        freqs_despues, mag_despues = espectro_fft(senial_120hz_despues, FS_NUEVA)
        graficar_resultados(
            t_120hz, senial_120hz_antes, senial_120hz_despues,
            freqs_antes, mag_antes, freqs_despues, mag_despues,
            FS_NUEVA, BANDA_INTERES, 'Ax', "ejercicio_3_filtrado_caida.png"
        )
        
        # --- EJERCICIO 4: Calcular Envolvente de Magnitud ---
        print("\n" + "="*50)
        print("COMIENZA EJERCICIO 4: CÁLCULO DE ENVOLVENTE")
        print("="*50)
        
        # 4.1: Remuestrear los 3 ejes
        print("  Paso 4.1: Remuestreando los 3 ejes a 120 Hz...")
        ax_120 = remuestrear_senal(Ax_500hz, FS_ORIGINAL, FS_NUEVA)
        ay_120 = remuestrear_senal(Ay_500hz, FS_ORIGINAL, FS_NUEVA)
        az_120 = remuestrear_senal(Az_500hz, FS_ORIGINAL, FS_NUEVA)
        # Asegurar que el vector de tiempo coincida (por si hay diff de 1 muestra)
        t_120hz_env = np.arange(len(ax_120)) / FS_NUEVA
        
        # 4.2: Aplicar filtro IIR a los 3 ejes (para limpiar ruido 50Hz)
        print("  Paso 4.2: Aplicando filtro IIR (Pasa-Banda) a los 3 ejes...")
        ax_120_f = aplicar_filtro(b_iir, a_iir, ax_120, usar_filfilt=True) # Fase cero
        ay_120_f = aplicar_filtro(b_iir, a_iir, ay_120, usar_filfilt=True)
        az_120_f = aplicar_filtro(b_iir, a_iir, az_120, usar_filfilt=True)
        
        # 4.3: Calcular envolvente (Magnitud -> Rectificar -> Filtro FIR)
        print("  Paso 4.3: Calculando envolvente (Magnitud + Rectif + Filtro FIR)...")
        rectificado, envolvente = calcular_envolvente(ax_120_f, ay_120_f, az_120_f, b_fir, a_fir)
        
        # 4.4: Graficar la envolvente
        print("  Paso 4.4: Generando gráfico de la envolvente...")
        plt.figure(figsize=(12, 6))
        plt.suptitle("Ejercicio 4: Envolvente de la Magnitud de Aceleración", fontsize=16)
        
        # Graficamos la magnitud rectificada (de donde viene)
        plt.plot(t_120hz_env, rectificado, label='Magnitud (sobre señal filtrada)', color='blue', alpha=0.6)
        # Graficamos la envolvente (resultado final)
        plt.plot(t_120hz_env, envolvente, label='Envolvente (LPF 1 Hz)', color='red', linewidth=2)
        
        plt.title(f'Envolvente vs. Magnitud (Fs = {FS_NUEVA} Hz)')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [g]')
        plt.legend()
        plt.grid(True)
        plt.xlim(t_120hz_env[0], t_120hz_env[-1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("ejercicio_4_envolvente_magnitud.png")
        plt.show()
        
        # Explicación Detallada PUNTO 5:
        # El Problema (Filtros FIR y Retardo):Los filtros FIR (como tu Pasa-Bajos de 1Hz) 
        # son de "fase lineal". Esto significa que introducen un retardo de grupo (group delay) 
        # constante. Este retardo es siempre la mitad del orden del filtro (N/2 muestras).Si hubieras
        # usado un filtro causal simple (como signal.lfilter), el pico de la envolvente (la línea roja)
        # habría aparecido retrasado en el tiempo.Si tu filtro FIR era de orden N=100, el retardo sería
        # N/2 = 50 muestras. A 120  Hz, eso es un error de 50 / 120 = 0.416 segundos. ¡Tu evento 
        # mal localizado!Tu Solución (Filtros de Fase Cero):Tú evitaste este problema correctamente. En 
        # tu FUNCIÓN 5 (aplicar_filtro), usaste usar_filfilt=True para la envolvente (en el Ej. 4, Paso 4.3)
        # .signal.filtfilt aplica el filtro una vez hacia adelante y luego otra vez hacia atrás.Este 
        # proceso (llamado "filtrado de fase cero") cancela por completo todo el retardo de grupo.Como 
        # resultado, el pico de tu envolvente (línea roja) queda perfectamente alineado en el tiempo con 
        # el pico de la magnitud (línea azul).En conclusión: El haber usado un filtro FIR podría haber 
        # afectado desastrosamente la localización del máximo, pero no lo hizo porque lo implementaste 
        # correctamente usando signal.filtfilt (filtrado de fase cero), lo que garantiza que la localización 
        # del pico es temporalmente precisa.
        
        #==================================================================
        # === INICIO EJERCICIO 5: LOCALIZACIÓN DEL EVENTO ===
        # ==================================================================
        
        print("\n" + "="*50)
        print("COMIENZA EJERCICIO 5: LOCALIZACIÓN DEL PICO")
        print("="*50)

        # 1. Encontrar el índice (la posición) del valor máximo en la envolvente
        #    np.argmax() nos da la posición del valor más alto
        indice_pico = np.argmax(envolvente)

        # 2. Usar ese índice para obtener los valores de tiempo y amplitud
        tiempo_pico = t_120hz_env[indice_pico]
        amplitud_pico = envolvente[indice_pico]

        print(f"  ¡Evento de caída detectado!")
        print(f"  Amplitud máxima de la envolvente: {amplitud_pico:.4f} [g]")
        print(f"  Tiempo de ocurrencia del pico: {tiempo_pico:.4f} [s]")

        # Opcional: Graficar el punto en el gráfico de la envolvente
        fig_env, ax_env = plt.subplots(1, 1, figsize=(12, 6))
        ax_env.plot(t_120hz_env, rectificado, label='Magnitud (sobre señal filtrada)', color='blue', alpha=0.6)
        ax_env.plot(t_120hz_env, envolvente, label='Envolvente (LPF 1 Hz)', color='red', linewidth=2)
        
        # --- Marcar el pico encontrado ---
        ax_env.plot(tiempo_pico, amplitud_pico, 'x', color='black', markersize=12, mew=3,
                    label=f'Pico detectado: {tiempo_pico:.2f}s')
        
        ax_env.set_title(f'Ejercicio 5: Localización del Pico del Evento (Fs = {FS_NUEVA} Hz)')
        ax_env.set_xlabel('Tiempo [s]')
        ax_env.set_ylabel('Amplitud [g]')
        ax_env.legend()
        ax_env.grid(True)
        ax_env.set_xlim(t_120hz_env[0], t_120hz_env[-1])
        plt.tight_layout()
        plt.savefig("ejercicio_5_pico_detectado.png")
        plt.show()

        print("\n--- Proceso completado exitosamente. ---")

    except Exception as e:
        # Captura cualquier error que haya ocurrido en los pasos
        print(f"\n--- ¡PROCESO FALLIDO! ---")
        print(f"Error: {e}")
        sys.exit(1) # Termina el script con un código de error