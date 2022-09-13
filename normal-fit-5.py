import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import pyplot as plt
from sklearn import linear_model
from datetime import datetime
import matplotlib.ticker as mtk

stock_file = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002003.csv'
# stock_file = '/Users/juns6831/work_root/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'

reg = linear_model.LinearRegression()

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily = daily.loc[:, ['open', 'close', 'high', 'low']]
# daily = daily.tail(500)

min, max = np.min(daily.close), np.max(daily.close)
daily['price_normal'] = (daily.close - min) / (max - min)
# daily['price_normal'] = (daily.close + daily.open) / 2
# daily['price_normal'] = daily.close
daily['price_y'] = None
daily['direction'] = None
daily['k'] = None

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

p = 10
i = p
while i < len(daily.index):
  s, e = i - p, i
  data = daily.iloc[s: e]

  train_X = np.array(range(p))
  train_Y = np.array(data.price_normal)
  reg.fit(train_X.reshape(-1, 1), train_Y)
  predict_Y = reg.predict(train_X.reshape(-1, 1))

  k = reg.coef_[0]
  model = f'model:{k}x+{reg.intercept_}'
  score = reg.score(train_X.reshape(-1, 1), train_Y)

  if score > 0.1 and -0.0003 < k < 0.0003:
    daily.iloc[s: e, 5] = predict_Y # price_y
    daily.iloc[s: e, 6] = 'up' # direction
    daily.iloc[s: e, 7] = k # k
    i = i + p
    # print(f'range: {s}-{e}')
    # print(f'k: {k}')
  else:
    i = i + 1

data1 = daily.iloc[0: i + p]
plot1 = plt.subplot(2, 1, 1)
plot2 = plt.subplot(2, 1, 2)
decorateAx(plot1, data1.index, data1.price_normal)
decorateAx(plot1, data1.index, data1.price_y)

# k_dis = np.quantile(np.array(ks), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# print(f'k dis: {k_dis}')
plot2.scatter(data1.index, data1.k)
plt.show()

