import pandas as pd
import mplfinance as mpf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model

stock_file = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'

reg = linear_model.LinearRegression()

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily = daily.loc[:, ['open', 'close', 'high', 'low']]
daily = daily.tail(100)

min, max = np.min(daily.close), np.max(daily.close)
daily['price_normal'] = (daily.close - min) / (max - min)

fig, axex = plt.subplots(3, 3)

for i in range(3):
  for j in range(3):
    data = daily.head((i + j + 1) * 10)
    reg.fit(daily.index.reshape(-1, 1), daily.price_normal.reshape(-1, 1))
    print(reg.coef_[0][0])
    predict_Y = reg.predict(daily.index.reshape(-1, 1))
    axex[i, j].plot(data.price_normal)
    axex[i, j].plot(predict_Y)

plt.show()