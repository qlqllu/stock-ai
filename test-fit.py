from sklearn import linear_model
reg = linear_model.LinearRegression()
model = reg.fit([[0, 0], [1, 1], [2, 2]], [1, 2, 4])
print(reg.coef_)
print(model.predict([[5, 5], [7, 7]]))
