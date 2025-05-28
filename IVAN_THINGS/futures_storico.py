# estrazione dei rolling dei futures
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Autenticazione su TradingView
username = 'alex2112545'
password = 'Gigino123.123'
tv = TvDatafeed()

# Simboli e exchange
symbol_data = {
    'HG_close': ('HG1!', 'COMEX'),          # rame
    'ALI_close': ('ALI1!', 'COMEX'),        # alluminio
    'BZ_close': ('BZN2025', 'NYMEX'),       # petrolio Brent
    'EUA_close': ('ECFZ2025', 'ICEENDEX')   # permessi CO₂
}

# Scaricare datistorici (1400 giorni, solo pr. chiusura)
dataframes = {}
for col_name, (symbol, exchange) in symbol_data.items():
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=1400)
    if df is not None and not df.empty:
        df = df[['close']].reset_index(drop=True) 
        dataframes[col_name] = df
    else:
        print(f"Errore: Dati non disponibili per {symbol} su {exchange}")


reference_asset = next(iter(dataframes)) 
reference_dates = tv.get_hist(
    symbol=symbol_data[reference_asset][0],
    exchange=symbol_data[reference_asset][1],
    interval=Interval.in_daily,
    n_bars=1400
).index.normalize()  # Date normalizzate

vix=tv.get_hist(symbol='VIX', exchange='CBOE', interval=Interval.in_daily, n_bars=1400)
vix = vix['close'].reset_index(drop=True)
vix.index = reference_dates[:len(vix)]
gold = tv.get_hist(symbol='GC1!', exchange='COMEX', interval=Interval.in_daily, n_bars=1400)
gold=gold['close'].reset_index(drop=True)
gold.index=reference_dates[:len(gold)]

# Combina i dati in un unico DataFrame
close_data = pd.concat(dataframes.values(), axis=1)
close_data.columns = list(dataframes.keys())  
close_data.index = reference_dates  

# Calcolo log-prezzi e rendimenti logaritmici
close_data_l = np.log(close_data)
returns = np.log(close_data / close_data.shift(1)).dropna()

len_train = int(len(returns)*0.8)                            #sta roba sta qua per praticita
returns_train = returns[:len_train]
returns_test = returns[len_train:]

signal_data=close_data['BZ_close'].values    #SELEZIONA L'ASSET DA UTILIZZARE COME SEGNALE
vix=vix.values
gold=gold.values


# Prepara i dati in un unico dizionario per comodità
plot_data = {
    'RAME': close_data['HG_close'],
    'Alluminio': close_data['ALI_close'],
    'Brent': close_data['BZ_close'],
    'EUA': close_data['EUA_close'],
    'VIX': pd.Series(vix, index=reference_dates[:len(vix)]),
    'Gold': pd.Series(gold, index=reference_dates[:len(gold)])
}

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
