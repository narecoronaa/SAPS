# -*- coding: utf-8 -*-
"""
Procesamiento de señal NIBP
- Lectura y visualización de registro a 1000 Hz
- Análisis espectral y determinación de requisitos de filtrado
- Comparación con respuesta del filtro analógico (Analog Filter Wizard / LTSpice)
- Prueba de filtros digitales (IIR y FIR) sobre señal remuestreada a 85 Hz
- Cálculo de MAP (presión arterial) y frecuencia cardíaca (LPM)
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from funciones_fft import fft_mag
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
import filter_parameters

plt.close('all')

# -----------------------
# Bloque 1: Lectura y graficación de la señal a 1000 Hz
# -----------------------
filename = 'nibp_1000hz.txt'
senial_1000 = np.loadtxt(filename)

FS = 1000.0                  # Frecuencia de muestreo original (Hz)
ts = 1.0 / FS
N = len(senial_1000)
t = np.linspace(0, N * ts, N)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(t, senial_1000)
ax.set_title('Señal NIBP (1000 Hz)')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Magnitud [mV]')
ax.grid(True)
plt.show()

# # # -----------------------
# # # Bloque 2: FFT de la señal 1000 Hz y búsqueda de picos relevantes
# # # -----------------------
frec, fft_senial = fft_mag(senial_1000, FS)

# 1. Convertir la magnitud de mV a Decibeles (dBmV)
#    Usamos 'with' para ignorar los errores de log(0)
with np.errstate(divide='ignore'):
    fft_db = 20 * np.log10(fft_senial)

# 2. Poner un "piso" de -100 dB a los valores -infinitos (que eran 0 mV)
fft_db[np.isneginf(fft_db)] = -100 

# 3. Graficar en escala Logarítmica (dB)
fig1, ax1 = plt.subplots(1, 1, figsize=(18, 12))
ax1.set_title('Tranformada de Fourier de la Senial (Escala Logarítmica)', fontsize=18)
ax1.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1.set_ylabel('Magnitud (dBmV)', fontsize=15) # <-- Etiqueta cambiada
ax1.grid(True, which="both")

# 4. Graficar la señal en dB
ax1.plot(frec, fft_db, label='FFT (dB)')

# 5. Aplicar zoom en los ejes X e Y para ver los detalles
ax1.set_xlim(0, 100) # Zoom de 0 a 100 Hz (para ver el ruido de 50 Hz)
ax1.set_ylim(-80, np.max(fft_db[1:]) + 10) # Zoom en Y (ignora el pico de DC)

# 6. Marcar las líneas críticas
ax1.axvline(x=20, color='g', linestyle='--', label='Banda de Interés (20 Hz)')
ax1.axvline(x=42.5, color='r', linestyle='--', label='F_Nyquist (42.5 Hz)')
ax1.axvline(x=50, color='m', linestyle='--', label='Ruido de Línea (50 Hz)')
ax1.legend()

plt.show()

# Determinación de interés: buscar máximo por encima de banda de paso (~20 Hz)
banda_pasante_max = 20.0   # [Hz] límite superior de la banda de interés
indices_validos = np.where(frec >= banda_pasante_max)[0]
if indices_validos.size > 0:
    magnitudes_filtradas = fft_senial[indices_validos]
    frecuencias_filtradas = frec[indices_validos]
    indice_max = np.argmax(magnitudes_filtradas)
    valor_max = magnitudes_filtradas[indice_max]
    frecuencia_max = frecuencias_filtradas[indice_max]
else:
    valor_max = 0.0
    frecuencia_max = np.nan


# Amplitud en Nyquist del sistema objetivo (fs/2 con fs=85 Hz -> nyq = 42.5 Hz)
FS_target = 85.0
indice_fs2 = int((FS_target / 2.0) * N / FS)  # índice aproximado en FFT
amp_fs_2 = fft_senial[indice_fs2]

print(f"Máximo (>= {banda_pasante_max} Hz): {valor_max:.3f} mV at {frecuencia_max:.3f} Hz")
print(f"Magnitud en fs/2 (={FS_target/2.0} Hz): {amp_fs_2:.3f} mV")


# -----------------------
# Bloque 3: Cálculo de resolución ADC y atenuaciones requeridas
# -----------------------
NUM_BITS = 16
VREF_mV = 5000.0   # Vref en mV (valor coherente con datos)
resolucion = VREF_mV / (2**NUM_BITS - 1)   # mV por LSB
print(f"Resolución ADC: {resolucion:.6f} mV/LSB")

# Atenuaciones (en dB) necesarias para que la componente indicada quede por debajo de 1 LSB
def attenuation_db(signal_mV, lsb_mV):
    # evita división por cero
    if lsb_mV <= 0 or signal_mV <= 0:
        return np.inf
    return 20.0 * np.log10(signal_mV / lsb_mV)

at_max = attenuation_db(valor_max, resolucion)
at_fs2_2 = attenuation_db(amp_fs_2, resolucion)

print(f"Atenuación requerida para reducir el pico {valor_max:.3f} mV a 1 LSB: {at_max:.2f} dB")
print(f"Atenuación requerida en fs/2 ({FS_target/2.0} Hz): {at_fs2_2:.2f} dB")

# -----------------------
# Bloque 4: Importar y comparar respuesta del filtro analógico (Analog Filter Wizard / LTSpice)
# -----------------------
# Se espera un CSV exportado por Analog Filter Wizard y un archivo de simulación LTspice
f_analog, mag_analog = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')
f_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')

# Localizar la frecuencia correspondiente a fs/2 del sistema objetivo dentro del eje del analógico
fs2_val = f_analog[np.where(f_analog >= (FS_target / 2.0))[0][0]]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xscale('log')
ax.plot(f_analog, mag_analog, label='Diseñado (Analog Filter Wizard)')
ax.plot(f_sim, mag_sim, label='Simulado (LTspice)')
ax.plot(fs2_val, at_fs2_2, 'X', label=f'Requisito en fs/2 ({FS_target/2.0} Hz)')
if not np.isnan(frecuencia_max):
    ax.plot(frecuencia_max, -at_max, 'o', label=f'Requisito en pico {frecuencia_max:.1f} Hz')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('|H(jω)| [dB]')
ax.grid(True, which='both')
ax.legend()
plt.show()

# -----------------------
# Bloque 5: Comentario sobre diseño digital (pyFDA)
# -----------------------
# (Diseños IIR/FIR fueron realizados en pyFDA; aquí se asumirá que los archivos .npz exportados están disponibles)
# - IIR para pulsos oscilométricos (1-11 Hz aprox.) con ripple <= 0.5 dB y alta atenuación en stopband.
# - FIR para componente lenta (presión en manguito) con passband hasta 0.5 Hz y atenuación >= 30 dB en frecuencias de oscilación.
# Los archivos .npz se importan en el siguiente bloque para pruebas.

# -----------------------
# Bloque 6: Prueba de filtros digitales sobre señal remuestreada a 85 Hz
# -----------------------
filename_85 = 'nibp_85hz.txt'
senial_85 = np.loadtxt(filename_85)

FS_resample = 85.0
ts_resample = 1.0 / FS_resample
N_resample = len(senial_85)
t_resample = np.linspace(0, N_resample * ts_resample, N_resample)

# Cargar filtros exportados desde pyFDA (.npz)
filtro_fir = np.load('filtro_fir.npz', allow_pickle=True)
filtro_iir = np.load('filtro_iir.npz', allow_pickle=True)

print("Parámetros del filtro IIR (report):")
filter_parameters.filter_parameters('filtro_iir.npz')
print("\nParámetros del filtro FIR (report):")
filter_parameters.filter_parameters('filtro_fir.npz')
print("\n")

# Extraer coeficientes (pyFDA guarda típicamente 'ba' o 'b')
Num_fir, Den_fir = filtro_fir['ba']
Num_iir, Den_iir = filtro_iir['ba']

# Espectros de los filtros (respuesta en frecuencia)
f_sig, sig_fft = fft_mag(senial_85, FS_resample)
f_fir, h_fir = signal.freqz(Num_fir, Den_fir, worN=f_sig, fs=FS_resample)
f_iir, h_iir = signal.freqz(Num_iir, Den_iir, worN=f_sig, fs=FS_resample)

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(f_fir, np.abs(h_fir), color='orange', label='FIR (mag)')
axs[0].legend()
axs[1].plot(f_iir, np.abs(h_iir), color='green', label='IIR (mag)')
axs[1].legend()
axs[1].set_xlim(0, 20)
axs[0].grid(); axs[1].grid()
plt.show()

# Aplicación causal de filtros: FIR (con padding para compensar transitorio) y IIR (lfilter)
# Para análisis offline se podría usar filtfilt para evitar desfase; aquí se aplica lfilter como en flujo causal.
# Ajuste del FIR: se añade padding de ceros equivalente a N/2 para compensar retardo y recortar.
ceros_agregar = filtro_fir['N'] // 2 if 'N' in filtro_fir else (len(Num_fir) - 1) // 2
senial_fir = np.pad(senial_85, (0, ceros_agregar), mode='constant')
senial_fir = signal.lfilter(Num_fir, Den_fir, senial_fir)
senial_fir = senial_fir[ceros_agregar:]

senial_iir = signal.lfilter(Num_iir, Den_iir, senial_85)

# Graficar espectros antes/después (normalizados)
f1_fir, senial_fir_fft = fft_mag(senial_fir, FS_resample)
f1_iir, senial_iir_fft = fft_mag(senial_iir, FS_resample)

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(f_sig, sig_fft / np.max(sig_fft), label='Original (norm)')
axs[0].plot(f1_fir, senial_fir_fft / np.max(sig_fft), label='FIR filtrada (norm)', color='red')
axs[0].legend(); axs[0].grid()
axs[1].plot(f_sig, sig_fft / np.max(sig_fft), label='Original (norm)')
axs[1].plot(f1_iir, senial_iir_fft / np.max(sig_fft), label='IIR filtrada (norm)', color='purple')
axs[1].legend(); axs[1].grid()
axs[1].set_xlim([0, FS_resample / 2.0])
plt.show()

# Graficar señales en tiempo (remuestreadas y filtradas)
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(t_resample, senial_fir, label='Señal filtrada - FIR')
axs[0].set_ylabel('mV'); axs[0].grid(); axs[0].legend()
axs[1].plot(t_resample, senial_iir, label='Señal filtrada - IIR')
axs[1].set_ylabel('mV'); axs[1].grid(); axs[1].legend()
axs[1].set_xlabel('Tiempo [s]')
plt.show()

# -----------------------
# Bloque 7: Cálculo de MAP y frecuencia cardíaca (LPM)
# - MAP: usar el instante del máximo de los pulsos oscilométricos sobre la señal filtrada IIR
# - FC: hallar la frecuencia dónde el espectro de los pulsos filtrados tiene su máximo y convertir a lpm
# -----------------------
idx_max_pulsos = np.argmax(senial_iir)
# convertir mV a mmHg con sensibilidad (usuario debe ajustar el factor de conversión)
SENSIBILIDAD_mV_por_mmHg = 18.0  # ejemplo (mV por mmHg)
map_mmHg = senial_fir[idx_max_pulsos] / SENSIBILIDAD_mV_por_mmHg

# Determinar frecuencia dominante en el espectro de la señal IIR (pulsos)
# buscar máximo en el espectro calculado anteriormente
freq_index = np.argmax(senial_iir_fft)
frec_maximo = f1_iir[freq_index]
fc_lpm = frec_maximo * 60.0

print(f"MAP estimada: {map_mmHg:.2f} mmHg (valor en tiempo de pico de pulsos)")
print(f"Frecuencia cardíaca estimada: {fc_lpm:.1f} lpm (frecuencia pico {frec_maximo:.3f} Hz)")

# -----------------------
# Fin del script
# -----------------------