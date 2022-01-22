"""
Main steps:
1. Train and a neural network that can identify the mode.
2. Createa another data set (must be different Et and k, different defects) to avoid data leakage
3. Use the trained model to predict the mode of new dataset.
4. Pick the data in new dataset that is in mode 1
"""

# import the library
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import sys

##% Data pre processing before training mode identification:
# firstly, load the data:
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\2level_defects.csv')

# identify extract the useful columns
# delete all the defect information except the defect class.
delete_col = ['Name', 'Et_eV_1', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'logk_1', 'bandgap_1', 'Et_eV_2', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2', 'logk_2', 'bandgap_2']
dfk = df.drop(delete_col, axis=1)

# encode the column: Mode.
dfk['Mode'] = pd.Categorical(dfk['Mode'])
dfk = pd.get_dummies(dfk)
# dfk.columns.values.tolist()
dfk = dfk.drop(['Mode_Two one-level'], axis=1)
# identify X and y
X_train = dfk.drop(['Mode_Single two-level'], axis=1)
y_train = dfk['Mode_Single two-level']

# scale the data:
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

# 1.Train a neural network that can identify the mode
# Use the best model: Neural network:
m_nn = MLPClassifier()
# define the parameters for grid search.
param_nn = {'hidden_layer_sizes':((100, 300, 300, 100), (100, 300, 500, 300, 100), (200, 600, 600, 200))}
# apply grid search:
grid_nn = GridSearchCV(m_nn, param_nn)
# train the model with the data.
grid_nn.fit(X_train_scaled, y_train)
