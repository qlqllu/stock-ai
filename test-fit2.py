import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

train_X = [0, 1, 2, 3, 4]
train_Y = [0, 2, 4, 6, 8]

train_X = np.array(train_X).reshape(-1, 1)
train_Y = np.array(train_Y).reshape(-1, 1)

test_X = np.array([5, 6, 7])

# 拟合
reg = LinearRegression()
reg.fit(train_X, train_Y)
a = reg.coef_[0][0]     # 系数
b = reg.intercept_[0]   # 截距
print('拟合的方程为：Y = %.6fX + %.6f' % (a, b))

prediction_y = reg.predict(test_X.reshape(-1, 1))

# 可视化
plt.figure('Title', figsize=(12,8))
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(train_X, train_Y, c='black')
plt.plot(test_X, prediction_y, c='r')
plt.show()