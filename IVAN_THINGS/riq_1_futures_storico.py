# Estrazione dei prezzi rolling dei futures principali + VIX + Gold
# Dati scaricati da TradingView usando tvDatafeed

import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval

# ==== CONFIGURAZIONE ====
N_BARS = 1400  # Numero di barre storiche (giornaliere) da scaricare

# ==== CONNESSIONE A TRADINGVIEW ====
tv = TvDatafeed()

# ==== SIMBOLI FUTURES ====
futures_symbols = {
    'HG_close': ('HG1!', 'COMEX'),        # Rame
    'ALI_close': ('ALI1!', 'COMEX'),      # Alluminio
    'BZ_close': ('BZN2025', 'NYMEX'),     # Brent
    'EUA_close': ('ECFZ2025', 'ICEENDEX') # Permessi CO₂
}

# ==== FUNZIONE PER SCARICARE SERIE DI CHIUSURA ====
def fetch_close_series(symbol, exchange, n_bars=N_BARS):
    df = tv.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=n_bars)
    if df is not None and not df.empty:
        return df['close'].reset_index(drop=True)
    else:
        print(f"❌ Errore: Dati non disponibili per {symbol} su {exchange}")
        return pd.Series(dtype=float)

# ==== SCARICA FUTURES PRINCIPALI ====
dataframes = {name: fetch_close_series(sym, exch) for name, (sym, exch) in futures_symbols.items()}

# ==== DEFINISCI DATE DI RIFERIMENTO ====
# Usa il primo asset disponibile come riferimento date
reference_asset = next(iter(dataframes))
reference_dates = tv.get_hist(futures_symbols[reference_asset][0], futures_symbols[reference_asset][1], interval=Interval.in_daily, n_bars=N_BARS).index.normalize()

# ==== SCARICA VIX E ORO ====
vix_series = fetch_close_series('VIX', 'CBOE', N_BARS)
gold_series = fetch_close_series('GC1!', 'COMEX', N_BARS)

# ==== COSTRUISCI DATAFRAME COMBINATO ====
close_data = pd.DataFrame({k: v for k, v in dataframes.items() if not v.empty})
close_data.index = reference_dates[:len(close_data)]

# ==== CALCOLA LOG-PREZZI E RENDIMENTI ====
log_prices = np.log(close_data)
returns = np.log(close_data / close_data.shift(1)).dropna()

# ==== SPLIT TRAIN/TEST (80/20) ====
split_point = int(len(returns) * 0.8)
returns_train = returns.iloc[:split_point]
returns_test = returns.iloc[split_point:]

# ==== PREPARA SERIE PER EXPORT O ANALISI ====
signal_data = close_data['BZ_close'].values  # Usa Brent come segnale
vix = vix_series.values
gold = gold_series.values

# ==== PREPARA DIZIONARIO PER PLOT ====
plot_data = {
    'RAME': close_data['HG_close'],
    'Alluminio': close_data['ALI_close'],
    'Brent': close_data['BZ_close'],
    'EUA': close_data['EUA_close'],
    'VIX': pd.Series(vix, index=reference_dates[:len(vix)]),
    'Gold': pd.Series(gold, index=reference_dates[:len(gold)])
}

# ==== STAMPA DIMENSIONI PER CONTROLLO ====
print("✅ Dimensioni serie scaricate:")
for name, series in plot_data.items():
    print(f"{name}: {series.shape}")

# ==== EXPORT VARIABILI PER USO ESTERNO ====
# (Se vuoi importarle da altri file)
__all__ = ['close_data', 'log_prices', 'returns', 'returns_train', 'returns_test', 'signal_data', 'vix', 'gold', 'plot_data']


# # Imposta la figura con 2 righe e 3 colonne
# fig, axs = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle('Serie Storiche Scaricate da TradingView', fontsize=16)

# # Ciclo per riempire ogni subplot
# for i, (key, series) in enumerate(plot_data.items()):
#     row, col = divmod(i, 3)  # calcola riga e colonna
#     axs[row, col].plot(series.index, series.values, label=key)
#     axs[row, col].set_title(key)
#     axs[row, col].set_xlabel('Data')
#     axs[row, col].set_ylabel('Prezzo')
#     axs[row, col].grid(True)
#     axs[row, col].legend()

# plt.tight_layout(rect=[0, 0, 1, 0.95])  # lascia spazio al titolo generale
# plt.show()

# # --- CREAZIONE DATAFRAME UNIFICATO PER EXPORT ---
# export_df = pd.DataFrame({
#     'Date': reference_dates
# })

# # Aggiungi ogni serie come colonna, allineando sulle date
# for key, series in plot_data.items():
#     export_df[key] = series.values

# # Imposta la colonna Date come indice
# export_df.set_index('Date', inplace=True)

# # --- SALVA IN EXCEL ---
# export_file = 'serie_storiche_tradingview.xlsx'
# export_df.to_excel(export_file)

# print(f"✅ File Excel esportato correttamente: {export_file}")
