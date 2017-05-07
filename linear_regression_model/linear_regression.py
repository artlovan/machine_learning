import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# suppress scipy warning
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# loading dataset
data_set = pd.read_csv('input_data.csv')
print(len(data_set))
print(data_set.columns.values)


# split data
def split_data(data_set):
    square_feet_values = []
    price_values = []
    for square_feet, price in zip(data_set['square_feet'], data_set['price']):
        square_feet_values.append([square_feet])
        price_values.append(price)
    return square_feet_values, price_values


train_x, train_y = split_data(data_set)
print(train_x)
print(train_y)


# build simple regression model
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

print('\n******************** linear_regression.py ********************\n\n')