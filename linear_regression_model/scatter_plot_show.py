from linear_regression_model.linear_regression import *

# lets see how we fit the model
plt.scatter(train_x, train_y, color='blue')
plt.plot(train_x, regr.predict(train_x), color='Red', linewidth=4)
plt.show()