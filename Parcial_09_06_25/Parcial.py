# -*- coding: utf-8 -*-
"""
Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Script para levantar registros de AUDIO (.wav), graficar señales temporales y sus espectros,
y calcular máximos espectrales y requerimientos para filtro antialiasing.

Autor: Narella Corona 
Fecha: Noviembre 2025
"""

#import process_data
import numpy as np
import matplotlib.pyplot as plt
#from pathlib import Path
#from funciones_fft import fft_mag
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
from scipy.io import wavfile
import filter_parameters
import process_code
from scipy import signal

def espectro_fft(y, sr):
    """Transformada rápida de Fourier con ventana de Hanning (suaviza espectro)."""
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    w = np.hanning(n)
    Y = np.fft.rfft(y * w)
    # Se normaliza la magnitud para obtener la amplitud correcta de la componente
    mag = np.abs(Y) / np.sum(w) 
    freqs = np.fft.rfftfreq(n, d=1/sr)
    return freqs, mag

def procesar_dataset(folder, FS=48000, T=3, guardar=True):
    
    archivos_audio = ['dataset_tos/tos1.wav', 'dataset_tos/tos2.wav', 'dataset_tos/tos3.wav', 'dataset_tos/tos4.wav']
    todas_las_seniales = []
    
    # Parámetros de la adquisición de diseño (16 bits, 3.3V)
    V_REF_ADC = 3.3
    N_BITS_ADC = 16
    DENOMINADOR = (2 ** N_BITS_ADC - 1)
    
    current_fs = FS # Usamos la Fs pasada como parámetro (48kHz)
    max_len = 0
    
    # 1. Lectura y escalado de datos de audio
    for archivo in archivos_audio:
        try:
            # Leer el archivo WAV: fs_wav = Frecuencia de muestreo, data_int = datos de la señal (enteros)
            fs_wav, data_int = wavfile.read(archivo)
            
            if fs_wav != current_fs:
                 # Esta Fs es la Fs de la señal de alta fidelidad, no la Fs de diseño.
                 current_fs = fs_wav
            
            # Escalar la señal a Voltios (se asume que la data es int16)
            senial_volt = data_int.astype(np.float64) * V_REF_ADC / DENOMINADOR
            todas_las_seniales.append(senial_volt)
            if len(senial_volt) > max_len:
                max_len = len(senial_volt)
        except FileNotFoundError:
            print(f"Error: Archivo '{archivo}' no encontrado. Saltando.")
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
    
    if not todas_las_seniales:
        print("No se cargaron señales de audio. Finalizando.")
        return

    # Creamos estructuras para la continuidad del script
    N = max_len
    num_capturas = len(todas_las_seniales)
    classmap = {i: f"Audio {i+1} ({archivos_audio[i]})" for i in range(num_capturas)}
    
    # Simulamos la estructura x, y, z del script original para el bucle:
    # x: matriz (num_capturas x N) de señales paddeadas en Voltios
    x_data = np.array([np.pad(s, (0, N - len(s))) for s in todas_las_seniales])
    
    # Vector de tiempo
    ts = 1 / current_fs
    t = np.linspace(0, N * ts, N, endpoint=False) 

    # --- Graficar temporal (Solo un canal por archivo) ---
    fig, axes = plt.subplots(num_capturas, 1, figsize=(15, 4 * num_capturas)) 
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_capturas):
        # La señal de audio ya está en Voltios
        x_volt = x_data[i] 
        gesture_name = i # Usamos el índice como 'gesture_name'
        
        # Manejo de un solo subplot vs múltiples subplots
        ax = axes[i] if num_capturas > 1 else axes 
        ax.plot(t, x_volt, label=classmap[gesture_name])
        
        ax.set_title(classmap[gesture_name] + " (Señal de Audio en [V])")
        ax.grid(); ax.legend(fontsize=6, loc='upper right')
        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Amplitud [V]')
        # El rango de la señal es [-3.3V, 3.3V] si usa todo el rango del ADC.
        ax.set_ylim(-V_REF_ADC, V_REF_ADC) 

    plt.tight_layout()
    plt.show()

    # --- Espectros y máximos ---
    fig_fft, axes_fft = plt.subplots(num_capturas, 1, figsize=(15, 4 * num_capturas)) 
    fig_fft.subplots_adjust(hspace=0.5)

    # FS de diseño para la adquisición (La consigna de diseño es 8 kHz)
    FS_DISENO = 8000    
    fs2_2 = FS_DISENO / 2 # Frecuencia de Nyquist/Rechazo del AAF: 4000 Hz

    # Inicializar diccionarios para guardar máximos (un solo eje 'A' de Audio)
    maximos = {i: {"A": {"valor": 0, "freq": 0}} for i in range(num_capturas)}
    maximos_fs2_2 = {i: {"A": {"valor": 0}} for i in range(num_capturas)}

    for i in range(num_capturas):
        x_volt = x_data[i]
        gesture_name = i
        
        # FFT de la señal usando la Fs original de los archivos (48kHz)
        freqs_x, mag_x = espectro_fft(x_volt, current_fs)
        
        ax_fft = axes_fft[i] if num_capturas > 1 else axes_fft
        ax_fft.plot(freqs_x, mag_x, label=classmap[gesture_name])

        # Análisis de máximos a partir de Nyquist del sistema de diseño (4 kHz)
        idx_x = np.where(freqs_x >= fs2_2)
        if len(idx_x[0]) > 0:
            h_max_x = np.max(mag_x[idx_x])
            f_max_x = freqs_x[idx_x][np.argmax(mag_x[idx_x])]
            maximos[gesture_name]["A"]["valor"] = h_max_x
            maximos[gesture_name]["A"]["freq"] = f_max_x

        # Análisis del valor exactamente en Nyquist del sistema de diseño (4 kHz)
        idx_x_fs2_2 = np.where(np.isclose(freqs_x, fs2_2))
        if len(idx_x_fs2_2[0]) > 0:
            h_fs2_2_x = np.max(mag_x[idx_x_fs2_2])
            maximos_fs2_2[gesture_name]["A"]["valor"] = h_fs2_2_x

        # Plotting Setup
        ax_fft.set_title(classmap[gesture_name] + " (Espectro en [V])")
        ax_fft.grid()
        ax_fft.legend(fontsize=6, loc='upper right')
        ax_fft.set_xlabel('Frecuencia [Hz]')
        ax_fft.set_ylabel('Magnitud [V]')
        # Usamos 10kHz como límite, un poco más allá de 8kHz (Nyquist de 4kHz)
        ax_fft.set_xlim(0, 10000) 
        
        # Agregar línea vertical en FS_DISENO/2
        ax_fft.axvline(x=fs2_2, color="black", linestyle="--", label=f'Nyquist de diseño ({fs2_2} Hz)')
        
        # Agregar marcas de los máximos
        max_val = maximos[gesture_name]["A"]["valor"]
        max_freq = maximos[gesture_name]["A"]["freq"]
        fs2_2_val = maximos_fs2_2[gesture_name]["A"]["valor"]
        
        if max_val > 0:
            ax_fft.plot(max_freq, max_val, marker='X', markersize=12, 
                          color='red', label=f'Máx. desde {fs2_2}Hz: {max_val*1000:.1f}mV')
        if fs2_2_val > 0:
            ax_fft.plot(fs2_2, fs2_2_val, marker='X', markersize=12, 
                          color='orange', label=f'Amplitud en {fs2_2}Hz: {fs2_2_val*1000:.1f}mV')


    plt.tight_layout()
    plt.show()

    # --- Resumen de resultados (Convertidos a mV para el AAF) ---
    print("\n--- Análisis de Requisitos del Filtro Antialiasing (AAF) ---")
    print(f"Sistema de Adquisición: FS = {FS_DISENO} Hz (Nyquist: {fs2_2} Hz), Resolución: {N_BITS_ADC} bits, Vref: {V_REF_ADC} V")

    # Mostrar máximos encontrados (Actualizar loop para 1 eje 'A')
    print("\nMáximos espectrales a partir de Nyquist de diseño (4kHz):")
    for gesto in maximos:
        print(f"Señal {classmap[gesto]}: Máximo={maximos[gesto]['A']['valor'] * 1000:.3f} mV en freq={maximos[gesto]['A']['freq']:.2f} Hz")
    print("\nMáximos exactamente en Nyquist de diseño (4kHz):")
    for gesto in maximos_fs2_2:
        print(f"Señal {classmap[gesto]}: Máximo={maximos_fs2_2[gesto]['A']['valor'] * 1000:.3f} mV en freq={fs2_2:.2f} Hz")

    # --- Determinar requerimientos del filtro antialiasing ---
    V_REF_mV = V_REF_ADC * 1000
    N_BITS = N_BITS_ADC
    RES = V_REF_mV/(2**N_BITS - 1)    # Resolución en mV
    
    # 5. Calcular peor caso de atenuación para el AAF (Ganancia requerida)
    peor_at_max = -np.inf
    peor_at_fs2_2 = -np.inf
    peor_gesto_max = ""
    peor_gesto_fs2_2 = ""

    for gesto in maximos:
        # Requerimiento de atenuación basado en la amplitud MÁXIMA que pasa la banda de rechazo
        h_max = maximos[gesto]["A"]['valor'] * 1000 # Amplitud pico de ruido/aliasing en mV
        at_max = 20 * np.log10(h_max / RES) if h_max > 0 else -np.inf
        if at_max > peor_at_max:
            peor_at_max = at_max
            peor_gesto_max = classmap[gesto]

        h_fs2_2 = maximos_fs2_2[gesto]["A"]['valor'] * 1000 # Amplitud pico en Nyquist en mV
        at_fs2_2 = 20 * np.log10(h_fs2_2 / RES) if h_fs2_2 > 0 else -np.inf
        if at_fs2_2 > peor_at_fs2_2:
            peor_at_fs2_2 = at_fs2_2
            peor_gesto_fs2_2 = classmap[gesto]

    print("\nREQUISITOS FINALES DEL FILTRO ANTIALIASING (AAF):")
    print(f"Atenuación MÍNIMA Requerida (Peor Caso en Banda de Rechazo, a partir de {fs2_2} Hz): {-peor_at_max:.2f} dB")
    print(f"Frecuencia de Rechazo: {fs2_2:.2f} Hz")
    
    # --- Análisis del filtro antialiasing ---
    try:
        # Importar resultados de Diseño de Analog Filter Wizard
        f_design, mag_design = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')
        
        # Importar resultados de simulación en LTSpice
        f_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')
        
        # Análisis de la atenuación del filtro simulado en las frecuencias de interés
        # NOTA: En dB, si la magnitud es < 1, el valor será negativo. La atenuación (A_req) es la magnitud NEGATIVA del peor_at_max
        at_req_dB = -peor_at_max
        
        # Encontrar la frecuencia del peor caso máximo
        freq_peor_caso = max(d['A']['freq'] for d in maximos.values())
        
        at1 = mag_sim[np.argmin(np.abs(f_sim-fs2_2))] 
        print(f"\nLa atenuación del filtro simulado en {fs2_2}Hz es de {at1:.2f}dB")
        
        if freq_peor_caso > 0:
            at2 = mag_sim[np.argmin(np.abs(f_sim-freq_peor_caso))] 
            print(f"La atenuación del filtro simulado en {freq_peor_caso:.2f}Hz es de {at2:.2f}dB")
        
        # Gráfica de comparación del filtro
        fig_filtro, ax_filtro = plt.subplots(1, 1, figsize=(12, 10))
        
        ax_filtro.set_title('Filtro Antialiasing - Análisis de Requisitos', fontsize=18)
        ax_filtro.set_xlabel('Frecuencia [Hz]', fontsize=15)
        ax_filtro.set_ylabel('|H(jω)|² [dB]', fontsize=15)
        #ax_filtro.set_xscale('log')
        ax_filtro.grid(True, which="both")
        
        # Plotear respuestas del filtro
        ax_filtro.plot(f_design, mag_design, label='Filtro Diseñado', linewidth=2)
        ax_filtro.plot(f_sim, mag_sim, label='Filtro Simulado', linewidth=2)
        
        # Marcar requisitos
        # El requisito de atenuación es -peor_at_max (la ganancia requerida para llevar el ruido al LSB)
        ax_filtro.plot(fs2_2, at_req_dB, marker='X', markersize=15, color='red',
                      label=f'Requisito en {fs2_2}Hz: {at_req_dB:.1f}dB')
        
        if freq_peor_caso > 0:
            ax_filtro.plot(freq_peor_caso, at_req_dB, marker='X', markersize=15, color='orange',
                          label=f'Requisito máximo: {at_req_dB:.1f}dB')
        
        
        ax_filtro.legend(loc="lower left", fontsize=12)
        ax_filtro.set_xlim(1, 10000) # Ajustado a la banda de audio
        ax_filtro.set_ylim(-100, 10) # Mayor atenuación
        
        plt.show()
        
    except Exception as e:
        # Se incluye el paso del AAF para que se ejecute la siguiente consigna
        print(f"\nNo se pudieron cargar los archivos del filtro. Asumo que es el siguiente punto del examen. Error: {e}")
        print("Continuando sin análisis del filtro...")
        
        

if __name__ == "__main__":
    # --- 1. FILTRADO ANALÓGICO (AAF) Y CÁLCULO DE REQUISITOS ---
    procesar_dataset("analisis_tos_examen", FS=48000)
    
    # -------------------------------------------------------------------------------------------------
    # --- 2. FILTRADO DIGITAL IIR (Prueba sobre una señal) ---
    # Consigna: Pruebe el filtro diseñado sobre una de las señales de prueba (remuestreado previamente a 8 kHz). 
    #           Grafique la señal y su espectro antes y después del filtrado.
    # -------------------------------------------------------------------------------------------------
    
    # Parámetros Fijos
    FS_DISENO = 8000 # Frecuencia de muestreo para el filtro IIR y remuestreo
    ARCHIVO_PRUEBA = 'dataset_tos/tos1.wav' 
    FILTRO_NPZ = 'Filtro_IIR_Pasabanda.npz'
    
    # Parámetros ADC para escalado (tomados de procesar_dataset)
    V_REF_ADC = 3.3
    N_BITS_ADC = 16
    DENOMINADOR = (2 ** N_BITS_ADC - 1)
    
    print("\n" + "="*50)
    print(f"COMIENZA PRUEBA DE FILTRO DIGITAL IIR (N=8) en {ARCHIVO_PRUEBA}")
    print("="*50)
    
    try:
        # A. Carga, Escalado y Remuestreo de la Señal
        
        # 1. Cargar la señal original (generalmente 48 kHz)
        fs_original, data_int = wavfile.read(ARCHIVO_PRUEBA)
        senial_original_volt = data_int.astype(np.float64) * V_REF_ADC / DENOMINADOR
        
        # 2. Remuestreo a 8 kHz
        N_original = len(senial_original_volt)
        N_resample = int(N_original * (FS_DISENO / fs_original))
        senial_remuestreada = signal.resample(senial_original_volt, N_resample)
        t_resampled = np.linspace(0, N_resample / FS_DISENO, N_resample, endpoint=False)
        
        # B. Carga y Aplicación del Filtro
        
        # 3. Cargar el filtro IIR Pasabanda
        filtro_iir = np.load(FILTRO_NPZ, allow_pickle=True)
        Num_iir, Den_iir = filtro_iir['ba']
        
        # 4. Aplicar el filtrado
        senial_filtrada = signal.lfilter(Num_iir, Den_iir, senial_remuestreada)
        
        # C. Cálculo y Graficación de Resultados
        
        # 5. Cálculo de Espectros (Fs = 8000 Hz)
        f_pre, mag_pre = espectro_fft(senial_remuestreada, FS_DISENO)
        f_post, mag_post = espectro_fft(senial_filtrada, FS_DISENO)
        
        # 6. Graficación (Dominio Temporal)
        fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig1.suptitle(f"Dominio Temporal: Filtrado IIR sobre {ARCHIVO_PRUEBA}", fontsize=16)
        
        ax1[0].plot(t_resampled, senial_remuestreada, label='Señal Original (8 kHz)', color='blue')
        ax1[0].set_title('Antes del Filtrado (Señal Remuestreada)')
        ax1[0].set_ylabel('Amplitud [V]', fontsize=12)
        ax1[0].grid()
        ax1[0].legend()
        
        ax1[1].plot(t_resampled, senial_filtrada, label='Señal Filtrada (IIR Pasabanda)', color='red')
        ax1[1].set_title('Después del Filtrado (500-1200 Hz)')
        ax1[1].set_xlabel('Tiempo [s]', fontsize=12)
        ax1[1].set_ylabel('Amplitud [V]', fontsize=12)
        ax1[1].grid()
        ax1[1].legend()
        
        plt.tight_layout()
        plt.show()

        # 7. Graficación (Dominio Frecuencial)
        fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig2.suptitle(f"Dominio Frecuencial: Espectro (Fs = {FS_DISENO} Hz)", fontsize=16)
        
        # Marcar la banda de interés para contexto
        BANDA_PASO = (500, 1200)
        
        ax2[0].plot(f_pre, mag_pre, label='Espectro Original (Remuestreado)', color='blue')
        ax2[0].axvspan(BANDA_PASO[0], BANDA_PASO[1], color='green', alpha=0.15, label='Banda de Paso IIR')
        ax2[0].set_title('Antes del Filtrado')
        ax2[0].set_ylabel('Magnitud [V]', fontsize=12)
        ax2[0].set_xlim([0, FS_DISENO / 2])
        ax2[0].grid()
        ax2[0].legend()

        ax2[1].plot(f_post, mag_post, label='Espectro Filtrado', color='red')
        ax2[1].axvspan(BANDA_PASO[0], BANDA_PASO[1], color='green', alpha=0.15)
        ax2[1].set_title('Después del Filtrado')
        ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=12)
        ax2[1].set_ylabel('Magnitud [V]', fontsize=12)
        ax2[1].set_xlim([0, FS_DISENO / 2])
        ax2[1].grid()
        ax2[1].legend()
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de prueba '{ARCHIVO_PRUEBA}' o el filtro '{FILTRO_NPZ}'. Asegúrese de que existen.")
    except Exception as e:
        print(f"Ocurrió un error inesperado en el filtrado digital: {e}")
        
    try:
        # -------------------------------------------------------------------------------------------------
        # --- 3. CÁLCULO DE ENVOLVENTE (CONSIGNA 4) ---
        # -------------------------------------------------------------------------------------------------
        FILTRO_FIR_NPZ = 'Filtro_FIR_Equiripple.npz' # Nombre del filtro FIR (Fc=20Hz)
        
        print("\n" + "="*50)
        print(f"COMIENZA CÁLCULO DE ENVOLVENTE (FIR) sobre la señal IIR")
        print("="*50)
    
        # A. Cargar el filtro FIR
        filtro_fir_env = np.load(FILTRO_FIR_NPZ, allow_pickle=True)
        Num_fir, Den_fir = filtro_fir_env['ba'] # Para FIR, Den_fir es [1.]
        
        # B. Rectificar la señal (Valor Absoluto)
        # Se usa la señal que ya pasó por el filtro IIR (senial_filtrada_IIR)
        senial_rectificada = np.abs(senial_filtrada) 
    
        # C. Aplicar el filtro FIR Pasa-Bajos (20 Hz)
        # Se usa lfilter para simular el procesamiento en tiempo real
        envolvente_fir = signal.lfilter(Num_fir, Den_fir, senial_rectificada)
    
        # D. Graficar la envolvente
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        fig3.suptitle("Cálculo de Envolvente (Consigna 4)", fontsize=16)
        
        # Grafica la señal base (Rectificada post-IIR)
        ax3.plot(t_resampled, senial_rectificada, label='Señal Rectificada (Post-IIR)', color='gray', alpha=0.5)
        # Grafica la envolvente (Resultado del FIR)
        ax3.plot(t_resampled, envolvente_fir, label='Envolvente (Post-FIR 20Hz)', color='red', linewidth=2)
        
        ax3.set_title(f"Envolvente (FIR LPF Fc=20Hz) sobre {ARCHIVO_PRUEBA}")
        ax3.set_xlabel('Tiempo [s]', fontsize=12)
        ax3.set_ylabel('Amplitud [V]', fontsize=12)
        ax3.grid()
        ax3.legend()
        ax3.set_xlim(t_resampled[0], t_resampled[-1])
        plt.show()
    
    except FileNotFoundError as e:
        print(f"ERROR: No se encontró un archivo de filtro necesario.")
        print(f"Detalle: {e.filename}")
    except Exception as e:
        print(f"Ocurrió un error inesperado en el filtrado digital: {e}")
    
    


    
    
