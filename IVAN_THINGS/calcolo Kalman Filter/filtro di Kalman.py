#Calcolare una serie di stato nascosto usando il filtro di Kalman sui dati vix e gold (con standardizzazione rolling),e preparare le variabili esogene per un modello (es. LSTM).

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


ROLLING_WINDOW = 649  #Fosssi in te non lo toccherei
vix_gold_df = pd.DataFrame({'vix': vix, 'gold': gold})
n_timesteps = len(vix_gold_df)
num_observed_variables = vix_gold_df.shape[1]

# Configura i parametri di Kalman
F = 1.0
H = np.ones((num_observed_variables, 1))
Q_val = 0.01                            #Cambiano la smoothness della serie di stato
R_generic_std = 0.5                     #Cambiano la smoothness della serie di stato
R = np.eye(num_observed_variables) * R_generic_std
