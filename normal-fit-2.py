# Can plot normalized price and line

import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import pyplot as plt
from sklearn import linear_model
from datetime import datetime
import matplotlib.ticker as mtk

stock_file = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'
# stock_file = '/Users/juns6831/work_root/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'

reg = linear_model.LinearRegression()

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily = daily.loc[:, ['open', 'close', 'high', 'low']]
# daily = daily.tail(1000)

min, max = np.min(daily.close), np.max(daily.close)
daily['price_normal'] = (daily.close - min) / (max - min)
# daily['dt'] = np.arange(start = 0, stop = len(daily), step = 1, dtype='int')
daily['price_y'] = None

def decorateAx(ax, xs, ys):
  def x_fmt_func(x, pos=None):
      idx = np.clip(int(x + 0.5), 0, len(xs) - 1)
      return xs[idx].strftime("%Y-%m-%d")

  idx_pxy = np.arange(len(xs))
  ax.plot(idx_pxy, ys, linewidth=1, linestyle="-")
  ax.plot(ax.get_xlim(), [0, 0], color="blue", linewidth=0.5, linestyle="--")
  ax.xaxis.set_major_formatter(mtk.FuncFormatter(x_fmt_func))
  ax.grid(True)
  return

plot1 = plt.subplot(1, 1, 1)
fig, axex = plt.subplots(3, 3)
p = 10
fig_num = 0
for i in range(0, len(daily.index) - p, p):
  data = daily.iloc[i: i + p]

  train_X = np.array(range(p))
  train_Y = np.array(data.price_normal)
  reg.fit(train_X.reshape(-1, 1), train_Y)
  predict_Y = reg.predict(train_X.reshape(-1, 1))

  k = reg.coef_[0]
  model = f'model:{k}x+{reg.intercept_}'
  score = reg.score(train_X.reshape(-1, 1), train_Y)

  if fig_num < 9:
    if score > 0.9:
      print(f'fig num: {fig_num}, k: {k}, score: {score}, between: {daily.index[i].strftime("%Y-%m-%d")} and {daily.index[i + p - 1].strftime("%Y-%m-%d")}')
      ax = axex[int(fig_num/3), fig_num%3]
      ax.set_title(f'{daily.index[i].strftime("%Y-%m-%d")} - {daily.index[i + p - 1].strftime("%Y-%m-%d")}')
      ax.plot(train_X, train_Y)
      ax.plot(train_X, predict_Y)
      fig_num += 1
      daily.iloc[i: (i + p), 5] = predict_Y
  else:
    break

data1 = daily.iloc[0:i + p]
# plot1.plot(data1.dt, data1.price_normal)
# plot1.plot(data1.dt, data1.price_y)
decorateAx(plot1, data1.index, data1.price_normal)
decorateAx(plot1, data1.index, data1.price_y)
plt.show()

