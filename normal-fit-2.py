import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model

# stock_file = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'
stock_file = '/Users/juns6831/work_root/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'

reg = linear_model.LinearRegression()

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily = daily.loc[:, ['open', 'close', 'high', 'low']]
# daily = daily.tail(1000)

min, max = np.min(daily.close), np.max(daily.close)
daily['price_normal'] = (daily.close - min) / (max - min)

fig, axex = plt.subplots(3, 3)

p = 10
fig_num = 0
for i in range(len(daily.index) - p):
  data = daily.iloc[i: i + p]

  train_X = np.array(range(p))
  train_Y = np.array(data.price_normal)
  reg.fit(train_X.reshape(-1, 1), train_Y)
  predict_Y = reg.predict(train_X.reshape(-1, 1))

  k = reg.coef_[0]
  model = f'model:{k}x+{reg.intercept_}'
  score = reg.score(train_X.reshape(-1, 1), train_Y)

  if score > 0.9 and fig_num < 9:
    # print(f'fig num: {fig_num}, model: {model}, score: {score}')
    print(f'fig num: {fig_num}, k: {k}, score: {score}, on {daily.index[i]}')
    ax = axex[int(fig_num/3), fig_num%3]
    ax.plot(train_X, train_Y)
    ax.plot(train_X, predict_Y)
    fig_num += 1

plt.show()