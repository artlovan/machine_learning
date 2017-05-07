from linear_regression_model.linear_regression import *
import pandas as pd
import numpy as np
from sklearn import linear_model

#  loading data
dataset = pd.read_csv('tv_show_viewers_datasest.csv')

print(len(dataset))
print(list(dataset.columns.values))

print('\n\n')
#  dataset statistics
print('******************************************** dataset statistics ********************************************\n')
print(dataset.describe())

print('\n\n')
#  top observations
print('******************************************** top observations ********************************************\n')
print(dataset.head())


#  split dataset
def split_data(dataset):
    features = []
    target = []
    for char1, char2, char3, char4, char5, fight_scene, comedy_scence, romance_scence, viewers_count in zip(
            dataset['Character1_appeared'],
            dataset['Character2_appeared'],
            dataset['Character3_appeared'],
            dataset['Character4_appeared'],
            dataset['Character5_appeared'],
            dataset['Fight_scenes'],
            dataset['Comedy_scences'],
            dataset['Romance_scence'],
            dataset['Viewers']
    ):
        features.append([char1, char2, char3, char4, char5, fight_scene, comedy_scence, romance_scence])
        target.append(viewers_count)
    return features, target


#  let's build linear regression model
print("********************************** let's build linear regression model ************************************\n")
train_features, train_target = split_data(dataset)
print(train_features)
print(train_target)

#  predicting viewers for new episode
regr = linear_model.LinearRegression()
regr.fit(train_features, train_target)

episode_51_features = np.array([4, 6, 3, 6, 3, 4, 8, 9]).reshape(1, -1)
print("********************************** episode 51 ************************************\n")
print(regr.predict(episode_51_features))

episode_52_features = np.array([4, 6, 3, 6, 3, 6, 2, 3]).reshape(1, -1)
print("********************************** episode 52 ************************************\n")
print(regr.predict(episode_52_features))