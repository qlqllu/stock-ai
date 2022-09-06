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
daily = daily.tail(1000)

min, max = np.min(daily.close), np.max(daily.close)
daily['price_normal'] = (daily.close - min) / (max - min)

fig, axex = plt.subplots(3, 3)

for i in range(3):
  for j in range(3):
    data = daily.iloc[(i + j) * 5: (i + j + 1) * 5]
    train_X = np.array(range(len(data.index)))
    train_Y = np.array(data.price_normal)
    reg.fit(train_X.reshape(-1, 1), train_Y)
    predict_Y = reg.predict(train_X.reshape(-1, 1))
    print(f'model:{reg.coef_[0]}x+{reg.intercept_}, score: {reg.score(train_X.reshape(-1, 1), train_Y)}')
    axex[i, j].plot(train_X, train_Y)
    axex[i, j].plot(train_X, predict_Y)

plt.show()