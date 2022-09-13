import pandas as pd
import mplfinance as mpf
import numpy as np

stock_file = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002003.csv'

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily = daily.loc[:, ['open', 'close', 'high', 'low']]

daily = daily.tail(100)

daily['color'] = daily.apply(lambda x: 'red' if (x.close - x.open) > 0 else 'green', axis=1)
# daily['price_change'] = daily.close - daily.close.shift(1)
# daily['price_change_pct'] = round((daily.close - daily.close.shift(1)) / daily.close.shift(1) * 100, 2)
min, max = np.min(daily.close), np.max(daily.close)
print(min, max)
daily['price_normal'] = (daily.close - min) / (max - min)

close_plot = mpf.make_addplot(daily.close,color='g')
price_change_pct_plog = mpf.make_addplot(daily.price_normal,color='r', panel = 1)

mpf.plot(daily, type='candle', marketcolor_overrides=daily['color'].values, addplot=[close_plot, price_change_pct_plog])