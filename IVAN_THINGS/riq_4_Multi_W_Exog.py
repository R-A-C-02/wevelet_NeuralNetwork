
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque

from riq_1_futures_storico  import signal_data
from riq_2_filtro_di_Kalman import kalman
from modwt import modwt, modwtmra  # Maximum Overlap Discrete Wavelet Transform + Multi-Resolution Analysis

# === LSTM MODELLO ===
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_units):
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])  # prendiamo solo l’ultimo step
        return self.output(out)

# === FUNZIONI ===
def multi_step_forecast(hist_main, hist_exog, model, scaler_main, scaler_exog, look_back, device, steps_ahead=10):
    norm_main = scaler_main.transform(np.array(hist_main).reshape(-1,1))
    norm_exog = scaler_exog.transform(np.array(hist_exog).reshape(-1,1))
    combined_seq = np.hstack([norm_main, norm_exog])

    input_tensor = torch.tensor(combined_seq[-look_back:].reshape(1, look_back, 2), dtype=torch.float32).to(device)
    model.eval()
    predictions = []
    exog_last = norm_exog[-1]

    with torch.no_grad():
        for _ in range(steps_ahead):
            out = model(input_tensor).cpu().numpy()
            predictions.append(scaler_main.inverse_transform(out)[0][0])
            new_step = np.array([[out[0][0], exog_last[0]]])
            combined_seq = np.vstack([combined_seq, new_step])
            input_tensor = torch.tensor(combined_seq[-look_back:].reshape(1, look_back, 2), dtype=torch.float32).to(device)

    return predictions

def retrain_model(hist_main, hist_exog, model, scaler_main, scaler_exog, look_back, device, epochs=5, batch_size=4):
    norm_main = scaler_main.fit_transform(np.array(hist_main).reshape(-1,1))
    norm_exog = scaler_exog.fit_transform(np.array(hist_exog).reshape(-1,1))
    combined_seq = np.hstack([norm_main, norm_exog])

    X, y = [], []
    for i in range(len(combined_seq) - look_back):
        X.append(combined_seq[i:i+look_back])
        y.append(combined_seq[i+look_back,0])

    if len(X) == 0:
        return

    X, y = np.array(X), np.array(y)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).reshape(-1,1).to(device)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in dataloader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

# === CONFIG ===
wavelet = "db7"
J = 5
look_back = 10
hidden_units = 100
test_len = 50
dwt_window = 417
start_idx = 650
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
scalers_main = {}
scalers_exog = {}
hist_main = {d: deque(maxlen=dwt_window) for d in range(J+1)}
hist_exog = deque(maxlen=dwt_window)
forecast_storage = {d: {1:[], 5:[], 10:[]} for d in range(J+1)}
train_preds = {d: {1:[], 5:[], 10:[]} for d in range(J+1)}

# === PREPARAZIONE STORICO ===
for t in range(start_idx - dwt_window, start_idx):
    if t < 0:
        for d in range(J+1): hist_main[d].append(0)
        hist_exog.append(0)
    else:
        coeffs = modwt(signal_data[max(0,t-dwt_window+1):t+1], wavelet, J)
        mra = modwtmra(coeffs, wavelet)
        for d in range(J+1):
            hist_main[d].append(mra[d][-1])
        hist_exog.append(kalman[t])

# === ALLENAMENTO INIZIALE ===
for d in range(J+1):
    main_arr = np.array(hist_main[d])
    exog_arr = np.array(hist_exog)
    scaler_main = StandardScaler().fit(main_arr.reshape(-1,1))
    scaler_exog = StandardScaler().fit(exog_arr.reshape(-1,1))
    norm_main = scaler_main.transform(main_arr.reshape(-1,1))
    norm_exog = scaler_exog.transform(exog_arr.reshape(-1,1))
    combined_seq = np.hstack([norm_main, norm_exog])

    X, y = [], []
    for i in range(len(combined_seq) - look_back):
        X.append(combined_seq[i:i+look_back])
        y.append(combined_seq[i+look_back,0])
    X, y = np.array(X), np.array(y)

    if len(X) > 0:
        model = SimpleLSTM(input_size=2, hidden_units=hidden_units).to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32).to(device),
                                            torch.tensor(y, dtype=torch.float32).reshape(-1,1).to(device))
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

        for _ in range(10):
            for xb, yb in dl:
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

        models[d] = model
        scalers_main[d] = scaler_main
        scalers_exog[d] = scaler_exog

        model.eval()
        with torch.no_grad():
            preds_all = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
            inv_preds = scaler_main.inverse_transform(preds_all)
            for h in [1,5,10]:
                train_preds[d][h].extend(inv_preds[:len(y)])

# === FORECAST E RETRAIN ===
for t in range(start_idx, start_idx + test_len):
    for d in range(J+1):
        preds = multi_step_forecast(list(hist_main[d]), list(hist_exog), models[d],
                                    scalers_main[d], scalers_exog[d], look_back, device)
        forecast_storage[d][1].append(preds[0])
        forecast_storage[d][5].append(preds[4])
        forecast_storage[d][10].append(preds[9])

        coeffs = modwt(signal_data[max(0,t-dwt_window+1):t+1], wavelet, J)
        mra = modwtmra(coeffs, wavelet)
        hist_main[d].append(mra[d][-1])
    hist_exog.append(kalman[t])

    if (t - start_idx) % 20 == 0:
        for d in range(J+1):
            retrain_model(list(hist_main[d]), list(hist_exog), models[d],
                          scalers_main[d], scalers_exog[d], look_back, device)

# === CALCOLO RESIDUI TRAIN ===
train_residuals = {d: {h: [] for h in [1, 5, 10]} for d in range(J + 1)}
for t in range(look_back, start_idx):
    coeffs = modwt(signal_data[max(0, t - dwt_window + 1):t + 1], wavelet, J)
    mra = modwtmra(coeffs, wavelet)
    for d in range(J + 1):
        actual = mra[d][-1]
        for h in [1, 5, 10]:
            idx = t - look_back
            if idx < len(train_preds[d][h]):
                train_residuals[d][h].append(actual - train_preds[d][h][idx])

# === RICOSTRUZIONE SEGNALE ===
forecasted_signals = {h: [] for h in [1,5,10]}
for idx in range(test_len):
    for h in [1,5,10]:
        vals = [forecast_storage[d][h][idx] if idx < len(forecast_storage[d][h]) else np.nan for d in range(J + 1)]
        forecasted_signals[h].append(np.nansum(vals))

# === PLOTTING ===
fig, axs = plt.subplots(2 + J + 1, 1, figsize=(14, 4 * (2 + J + 1)))
fig.suptitle("Dashboard Forecast & Residui", fontsize=18)

axs[0].plot(range(start_idx, start_idx + test_len), signal_data[start_idx:start_idx + test_len], label="Segnale Reale", linewidth=2)
for h, style in zip([1, 5, 10], ['--', '-.', ':']):
    axs[0].plot(range(start_idx, start_idx + test_len), forecasted_signals[h], label=f"Forecast t+{h}", linestyle=style)
axs[0].set_title("Confronto Segnale Previsto vs Reale")
axs[0].legend()
axs[0].grid()

axs[1].plot(range(start_idx, start_idx + test_len), signal_data[start_idx:start_idx + test_len], label="True Signal", linewidth=2)
for h, style in zip([1, 5, 10], ['--', '-.', ':']):
    axs[1].plot(range(start_idx, start_idx + test_len), forecasted_signals[h], label=f"Forecast t+{h}", linestyle=style)
axs[1].set_title("Confronto Segnale Forecasted Ricostruito")
axs[1].legend()
axs[1].grid()

print("\n──────── METRICHE DI CONFRONTO ────────")
for h in [1, 5, 10]:
    true = signal_data[start_idx:start_idx + test_len]
    pred = np.array(forecasted_signals[h])
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - true))
    mape = np.mean(np.abs((pred - true) / true)) * 100
    print(f"Orizzonte t+{h}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

for d in range(J + 1):
    ax = axs[2 + d]
    for h, style in zip([1, 5, 10], ['--', '-.', ':']):
        ax.plot(range(look_back, look_back + len(train_residuals[d][h])),
                train_residuals[d][h], linestyle=style, label=f"Residui t+{h}")
    ax.set_title(f"Residui Train - Livello D{d}")
    ax.legend()
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
