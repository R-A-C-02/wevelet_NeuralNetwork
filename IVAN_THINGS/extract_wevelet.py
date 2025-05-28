** riqaudro 2**

'''
vix=tv.get_hist(symbol='VIX', exchange='CBOE', interval=Interval.in_daily, n_bars=1400)

ROLLING_WINDOW = 649  #Fosssi in te non lo toccherei

n_timesteps = len(vix_gold_df)

for t in range(n_timesteps):
    start_idx = max(0, t - ROLLING_WINDOW)

'''



**riquadro 3**

start_idx = 650



**riquadro 4**
start_idx = 650

**riquadro 5**
