import matplotlib.pyplot as plt
from modwt import modwt, modwtmra
from riq_1_futures_storico  import signal_data
from riq_4_Multi_W_Exog import wavelet, J, test_len, dwt_window, start_idx

# === SCOPO ===
# Verificare se i coefficienti wavelet (DWT) sono continui
# prima e dopo lo start_idx → per controllare la qualità della decomposizione


# Prepariamo i contenitori per i coefficienti
coeffs_before = {level: [] for level in range(J + 1)}  # prima di start_idx
coeffs_after = {level: [] for level in range(J + 1)}   # dopo start_idx

# --- Calcolo dei coefficienti PRIMA di start_idx ---
for t in range(start_idx - dwt_window, start_idx):
    signal_slice = signal_data[max(0, t - dwt_window + 1):t + 1]
    coeffs = modwt(signal_slice, wavelet, J)
    mra = modwtmra(coeffs, wavelet)
    for level in range(J + 1):
        last_value = mra[level][-1] if len(mra[level]) > 0 else 0
        coeffs_before[level].append(last_value)

# --- Calcolo dei coefficienti DOPO start_idx ---
for t in range(start_idx, start_idx + test_len):
    signal_slice = signal_data[max(0, t - dwt_window + 1):t + 1]
    coeffs = modwt(signal_slice, wavelet, J)
    mra = modwtmra(coeffs, wavelet)
    for level in range(J + 1):
        last_value = mra[level][-1] if len(mra[level]) > 0 else 0
        coeffs_after[level].append(last_value)

# === PLOTTING DEI RISULTATI ===
plt.figure(figsize=(14, 8))
for level in range(J + 1):
    plt.subplot(J + 1, 1, level + 1)
    
    # Coefficienti prima di start_idx (blu)
    plt.plot(range(start_idx - dwt_window, start_idx),
             coeffs_before[level],
             label=f"Livello {level} Prima", color="blue")
    
    # Coefficienti dopo start_idx (rosso)
    plt.plot(range(start_idx, start_idx + test_len),
             coeffs_after[level],
             label=f"Livello {level} Dopo", color="red")
    
    plt.axvline(start_idx, color='black', linestyle="--", label="Start Index")
    plt.title(f"Coefficiente Wavelet - Livello {level}")
    plt.xlabel("Tempo (t)")
    plt.ylabel(f"D{level}(t)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
