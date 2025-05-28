#Calcolare una serie di stato nascosto usando il filtro di Kalman sui dati vix e gold (con standardizzazione rolling),e preparare le variabili esogene per un modello (es. LSTM).

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# ==== IMPORTA I DATI DAL FILE ====
from riq_1_futures_storico import vix, gold

# ==== CONFIGURAZIONE ====
ROLLING_WINDOW = 649

# Prepara il DataFrame
vix_gold_df = pd.DataFrame({'vix': vix, 'gold': gold})
n_timesteps, num_observed_variables = vix_gold_df.shape


# Configura i parametri di Kalman
F = 1.0
H = np.ones((num_observed_variables, 1))
Q_val = 0.01                            #Cambiano la smoothness della serie di stato
R_generic_std = 0.5                     #Cambiano la smoothness della serie di stato
R = np.eye(num_observed_variables) * R_generic_std

# Inizializzazione Kalman
x_hat_t_minus_1 = np.array([[0]])        # stato stimato iniziale
P_t_minus_1 = np.array([[1]])            # varianza stimata iniziale
kalman_series = np.zeros(n_timesteps)

# ==== FILTRO DI KALMAN ====
for t in range(n_timesteps):
    start_idx = max(0, t - ROLLING_WINDOW)
    window_data = vix_gold_df.iloc[start_idx:t+1].values

    scaler = StandardScaler()
    scaler.fit(window_data)
    z_t = scaler.transform(window_data)[-1].reshape(num_observed_variables, 1)

    # Predizione Kalman
    x_hat_minus_t = F * x_hat_t_minus_1
    P_minus_t = F * P_t_minus_1 * F + Q_val

    # Aggiornamento Kalman
    y_t = z_t - H @ x_hat_minus_t
    S_t = H @ P_minus_t @ H.T + R
    K_t = P_minus_t @ H.T @ np.linalg.inv(S_t)
    x_hat_t = x_hat_minus_t + K_t @ y_t
    P_t = (np.eye(1) - K_t @ H) @ P_minus_t

    kalman_series[t] = x_hat_t.item()
    x_hat_t_minus_1, P_t_minus_1 = x_hat_t, P_t

# Standardizzazione della serie di stato
kalman = kalman_series

print(f"Serie dello stato nascosto 'kalman' calcolata da {num_observed_variables} serie con scaler rolling finestra={ROLLING_WINDOW}.")

# Plot della serie dello stato nascosto
plt.figure(figsize=(12, 6))
plt.plot(kalman, label='Kalman Filter State (rolling scale)', color='purple')
plt.title('Stato Nascosto Stimato (Rolling Scaling)')
plt.legend()
plt.grid(True)
plt.show()