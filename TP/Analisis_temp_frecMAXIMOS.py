# -*- coding: utf-8 -*-
"""
Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Script para levantar registros desde .csv, graficar señales temporales y sus espectros,
y calcular máximos espectrales y requerimientos para filtro antialiasing.

Autor: Albano Peñalva + adaptado por Copilot
Fecha: Septiembre 2025
"""

import process_data
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from funciones_fft import fft_mag
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard

def espectro_fft(y, sr):
    """Transformada rápida de Fourier con ventana de Hanning (suaviza espectro)."""
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    w = np.hanning(n)
    Y = np.fft.rfft(y * w)
    mag = np.abs(Y) / np.sum(w)
    freqs = np.fft.rfftfreq(n, d=1/sr)
    return freqs, mag

def procesar_dataset(folder, FS=500, T=3, guardar=True):
    output_dir = Path(__file__).parent

    # Lectura de datos
    x, y, z, classmap = process_data.process_data(FS, T, folder)

    ts = 1 / FS
    N = FS * T
    t = np.linspace(0, N * ts, N)

    fig, axes = plt.subplots(len(classmap), 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)

    # Graficar temporal
    trial_num = 0
    for gesture_name in classmap:
        for capture in range(int(len(x))):
            if (x[capture, N] == gesture_name):
                x_volt = x[capture, 0:N]*300 + 1650
                y_volt = y[capture, 0:N]*300 + 1650
                z_volt = z[capture, 0:N]*300 + 1650

                axes[gesture_name][0].plot(t, x_volt, label=f"Trial {trial_num}")
                axes[gesture_name][1].plot(t, y_volt, label=f"Trial {trial_num}")
                axes[gesture_name][2].plot(t, z_volt, label=f"Trial {trial_num}")
                trial_num += 1

        for i, eje in enumerate(["X", "Y", "Z"]):
            axes[gesture_name][i].set_title(classmap[gesture_name] + f" (Aceleración {eje} en [V])")
            axes[gesture_name][i].grid(); axes[gesture_name][i].legend(fontsize=6, loc='upper right')
            axes[gesture_name][i].set_xlabel('Tiempo [s]')
            axes[gesture_name][i].set_ylabel('Aceleración [V]')
            axes[gesture_name][i].set_ylim(-6*300+1650, 6*300+1650)

    plt.tight_layout()
    if guardar:
        fig.savefig(output_dir / f"{folder}_seniales_temporales.png", dpi=150)
    plt.show()

    # --- Espectros y máximos ---
    fig_fft, axes_fft = plt.subplots(len(classmap), 3, figsize=(20, 20))
    fig_fft.subplots_adjust(hspace=0.5)
    trial_num = 0

    FS2 = 60   # Nueva frecuencia de muestreo propuesta
    fs2_2 = FS2 / 2

    # Inicializar diccionarios para guardar máximos
    maximos = {gesture_name: {"X": {"valor": 0, "freq": 0},
                             "Y": {"valor": 0, "freq": 0},
                             "Z": {"valor": 0, "freq": 0}}
               for gesture_name in classmap}

    maximos_fs2_2 = {gesture_name: {"X": {"valor": 0},
                                   "Y": {"valor": 0},
                                   "Z": {"valor": 0}}
                     for gesture_name in classmap}

    for gesture_name in classmap:
        for capture in range(int(len(x))):
            if (x[capture, N] == gesture_name):
                x_volt = x[capture, 0:N]*300 + 1650
                y_volt = y[capture, 0:N]*300 + 1650
                z_volt = z[capture, 0:N]*300 + 1650

                # FFT X
                freqs_x, mag_x = espectro_fft(x_volt, FS)
                axes_fft[gesture_name][0].plot(freqs_x, mag_x, label=f"Trial {trial_num}")

                idx_x = np.where(freqs_x >= fs2_2)
                if len(idx_x[0]) > 0:
                    h_max_x = np.max(mag_x[idx_x])
                    f_max_x = freqs_x[idx_x][np.argmax(mag_x[idx_x])]
                    if h_max_x > maximos[gesture_name]["X"]["valor"]:
                        maximos[gesture_name]["X"]["valor"] = h_max_x
                        maximos[gesture_name]["X"]["freq"] = f_max_x

                idx_x_fs2_2 = np.where(freqs_x == fs2_2)
                if len(idx_x_fs2_2[0]) > 0:
                    h_fs2_2_x = np.max(mag_x[idx_x_fs2_2])
                    if h_fs2_2_x > maximos_fs2_2[gesture_name]["X"]["valor"]:
                        maximos_fs2_2[gesture_name]["X"]["valor"] = h_fs2_2_x

                # FFT Y
                freqs_y, mag_y = espectro_fft(y_volt, FS)
                axes_fft[gesture_name][1].plot(freqs_y, mag_y, label=f"Trial {trial_num}")

                idx_y = np.where(freqs_y >= fs2_2)
                if len(idx_y[0]) > 0:
                    h_max_y = np.max(mag_y[idx_y])
                    f_max_y = freqs_y[idx_y][np.argmax(mag_y[idx_y])]
                    if h_max_y > maximos[gesture_name]["Y"]["valor"]:
                        maximos[gesture_name]["Y"]["valor"] = h_max_y
                        maximos[gesture_name]["Y"]["freq"] = f_max_y

                idx_y_fs2_2 = np.where(freqs_y == fs2_2)
                if len(idx_y_fs2_2[0]) > 0:
                    h_fs2_2_y = np.max(mag_y[idx_y_fs2_2])
                    if h_fs2_2_y > maximos_fs2_2[gesture_name]["Y"]["valor"]:
                        maximos_fs2_2[gesture_name]["Y"]["valor"] = h_fs2_2_y

                # FFT Z
                freqs_z, mag_z = espectro_fft(z_volt, FS)
                axes_fft[gesture_name][2].plot(freqs_z, mag_z, label=f"Trial {trial_num}")

                idx_z = np.where(freqs_z >= fs2_2)
                if len(idx_z[0]) > 0:
                    h_max_z = np.max(mag_z[idx_z])
                    f_max_z = freqs_z[idx_z][np.argmax(mag_z[idx_z])]
                    if h_max_z > maximos[gesture_name]["Z"]["valor"]:
                        maximos[gesture_name]["Z"]["valor"] = h_max_z
                        maximos[gesture_name]["Z"]["freq"] = f_max_z

                idx_z_fs2_2 = np.where(freqs_z == fs2_2)
                if len(idx_z_fs2_2[0]) > 0:
                    h_fs2_2_z = np.max(mag_z[idx_z_fs2_2])
                    if h_fs2_2_z > maximos_fs2_2[gesture_name]["Z"]["valor"]:
                        maximos_fs2_2[gesture_name]["Z"]["valor"] = h_fs2_2_z

                trial_num += 1

        for i, eje in enumerate(["X", "Y", "Z"]):
            axes_fft[gesture_name][i].set_title(classmap[gesture_name] + f" (FFT {eje} en [mV])")
            axes_fft[gesture_name][i].grid()
            axes_fft[gesture_name][i].legend(fontsize=6, loc='upper right')
            axes_fft[gesture_name][i].set_xlabel('Frecuencia [Hz]')
            axes_fft[gesture_name][i].set_ylabel('Magnitud')
            axes_fft[gesture_name][i].set_xlim(0, 100)  # Ampliamos para ver las cruces
            
            # Agregar línea vertical en FS2/2
            axes_fft[gesture_name][i].axvline(x=fs2_2, color="black", linestyle="--", label='FS2/2')
            
            # Agregar marcas de los máximos para este gesto y eje
            if eje == "X":
                max_val = maximos[gesture_name]["X"]["valor"]
                max_freq = maximos[gesture_name]["X"]["freq"]
                fs2_2_val = maximos_fs2_2[gesture_name]["X"]["valor"]
            elif eje == "Y":
                max_val = maximos[gesture_name]["Y"]["valor"]
                max_freq = maximos[gesture_name]["Y"]["freq"]
                fs2_2_val = maximos_fs2_2[gesture_name]["Y"]["valor"]
            else:  # Z
                max_val = maximos[gesture_name]["Z"]["valor"]
                max_freq = maximos[gesture_name]["Z"]["freq"]
                fs2_2_val = maximos_fs2_2[gesture_name]["Z"]["valor"]
            
            if max_val > 0:
                axes_fft[gesture_name][i].plot(max_freq, max_val, marker='X', markersize=12, 
                                             color='red', label=f'Máximo desde FS2/2: {max_val:.1f}mV')
            if fs2_2_val > 0:
                axes_fft[gesture_name][i].plot(fs2_2, fs2_2_val, marker='X', markersize=12, 
                                             color='orange', label=f'Amplitud en FS2/2: {fs2_2_val:.1f}mV')

    plt.tight_layout()
    if guardar:
        fig_fft.savefig(output_dir / f"{folder}_espectros.png", dpi=150)
    #plt.show()

    # Mostrar máximos encontrados
    print("Máximos espectrales desde FS/2 por gesto y eje:")
    for gesto in maximos:
        for eje in maximos[gesto]:
            print(f"Gesto {classmap[gesto]}, Eje {eje}: Máximo={maximos[gesto][eje]['valor']:.3f} en freq={maximos[gesto][eje]['freq']:.2f} Hz")
    print("\nMáximos exactamente en FS/2 por gesto y eje:")
    for gesto in maximos_fs2_2:
        for eje in maximos_fs2_2[gesto]:
            print(f"Gesto {classmap[gesto]}, Eje {eje}: Máximo={maximos_fs2_2[gesto][eje]['valor']:.3f} en freq={fs2_2:.2f} Hz")

    # --- Determinar requerimientos del filtro antialiasing ---
    V_REF = 3300
    N_BITS = 12
    RES = V_REF/(2**N_BITS - 1)    # Resolución en mV

    peor_at_max = -np.inf
    peor_at_fs2_2 = -np.inf
    peor_gesto_max = ""
    peor_eje_max = ""
    peor_gesto_fs2_2 = ""
    peor_eje_fs2_2 = ""

    for gesto in maximos:
        for eje in maximos[gesto]:
            h_max = maximos[gesto][eje]['valor']
            at_max = 20 * np.log10(h_max / RES) if h_max > 0 else -np.inf
            if at_max > peor_at_max:
                peor_at_max = at_max
                peor_gesto_max = classmap[gesto]
                peor_eje_max = eje

            h_fs2_2 = maximos_fs2_2[gesto][eje]['valor']
            at_fs2_2 = 20 * np.log10(h_fs2_2 / RES) if h_fs2_2 > 0 else -np.inf
            if at_fs2_2 > peor_at_fs2_2:
                peor_at_fs2_2 = at_fs2_2
                peor_gesto_fs2_2 = classmap[gesto]
                peor_eje_fs2_2 = eje

    # print("\nResumen de requerimientos para filtro antialiasing:")
    # print(f"Peor caso de atenuación a partir de FS2/2: {peor_at_max:.2f} dB "
    #       f"(Gesto: {peor_gesto_max}, Eje: {peor_eje_max})")
    # print(f"Peor caso de atenuación exactamente en FS2/2: {peor_at_fs2_2:.2f} dB "
    #       f"(Gesto: {peor_gesto_fs2_2}, Eje: {peor_eje_fs2_2})")
    # print(f"Banda de paso recomendada: hasta 15 Hz")
    # print(f"Banda de rechazo: desde {fs2_2:.2f} Hz (FS2/2)")
    # print(f"Diseñar filtro para atenuar al menos {peor_at_max:.2f} dB en banda de rechazo.")

    # --- Análisis del filtro antialiasing ---
    try:
        # Importar resultados de Diseño de Analog Filter Wizard
        f_design, mag_design = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')
        
        # Importar resultados de simulación en LTSpice
        f_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')
        
        # Análisis de la atenuación del filtro simulado en las frecuencias de interés
        at1 = mag_sim[np.argmin(np.abs(f_sim-fs2_2))] 
        print(f"\nLa atenuación del filtro simulado en {fs2_2}Hz es de {at1:.2f}dB")
        
        # Encontrar la frecuencia del peor caso máximo
        freq_peor_caso = 0
        for gesto in maximos:
            for eje in maximos[gesto]:
                if maximos[gesto][eje]['freq'] > freq_peor_caso:
                    freq_peor_caso = maximos[gesto][eje]['freq']
        
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
        ax_filtro.plot(fs2_2, -peor_at_fs2_2, marker='X', markersize=15, color='red',
                      label=f'Requisito en FS2/2: {-peor_at_fs2_2:.1f}dB')
        
        if freq_peor_caso > 0:
            ax_filtro.plot(freq_peor_caso, -peor_at_max, marker='X', markersize=15, color='orange',
                          label=f'Requisito máximo: {-peor_at_max:.1f}dB')
        
        # Respuesta del filtro implementado
        f_impl = [1, 5, 7, 10, 12, 15, 20, 23, 26, 27, 29, 30, 31, 33, 35, 40, 45, 50, 55, 60, 70]
        mag_impl_volt = [2.24, 2.24, 2.36, 2.52, 2.44, 1.96, 1.18, 0.92, 0.7, 0.64, 0.58, 0.52, 0.48, 0.4, 0.38, 0.32, 0.24, 0.17, 0.15, 0.12, 0.084]
        # Convertir a dB: 20*log10(ganancia lineal=Vout/Vin)
        mag_impl = [20 * np.log10(g/2.24) for g in mag_impl_volt]
        ax_filtro.plot(f_impl, mag_impl, 'o-', label='Filtro Implementado', linewidth=2)
        
        
        ax_filtro.legend(loc="lower left", fontsize=12)
        ax_filtro.set_xlim(1, 100)
        ax_filtro.set_ylim(-35, 10)
        
        if guardar:
            fig_filtro.savefig(output_dir / f"{folder}_filtro_antialiasing.png", dpi=150)
        plt.show()
        
    except Exception as e:
        print(f"\nNo se pudieron cargar los archivos del filtro: {e}")
        print("Continuando sin análisis del filtro...")

if __name__ == "__main__":
    procesar_dataset("dataset_voley", FS=500, T=3)