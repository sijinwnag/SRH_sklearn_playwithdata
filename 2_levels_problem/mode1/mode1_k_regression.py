"""
This file do k regression 2 level defects for k_1, Mode: Two one-level.
"""

################################################################################
# import the library
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import sys

# import the function file from another folder:
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem')
from function_for_trainings import regression_repeat, regression_training

################################################################################
# data pre processing:

# load the data.
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\2level_defects.csv')
# df.head()
# identify extract the useful columns
# Drop first column of dataframe
df = df.iloc[: , 1:]
delete_col = delete_col = ['Name', 'Et_eV_1', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'bandgap_1', 'Et_eV_2', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2', 'bandgap_2']
dfk = df.drop(delete_col, axis=1)

# extract thet ones that are in mode
dfk = dfk[dfk['Mode']=='Two one-level']
dfk = dfk.drop(['Mode'], axis=1)
# define X and y
X = dfk.drop(['logk_1', 'logk_2'], axis=1)
y = dfk['logk_1'] + dfk['logk_2']
X = np.log(X)
# X.head()
# send it to regression repeat to train and evaluate the model.
r2_frame_sum = regression_repeat(X, y, 1, plot=True)
# plan:
# 1. plot the graph see what went wrong (done)
# 2. Neural network increase complexity (done)
# 3. play with 2 one level defects equation: see if there is any hint your can get.
# 4. Maybe you should do regression for both k1 and k2 together: do regression for sum of logk first then do the regression for difference of log k

# let the program know the logk1 + logk2:
X['logk sum'] = dfk['logk_1'] + dfk['logk_2']
# try y2 be the difference of the log k1 and log k2
y2 = dfk['logk_1'] - dfk['logk_2']
r2_frame_diff = regression_repeat(X, y2, 1, plot=True)

# Thus log k1 and log k2 can be calcualted saperately

# %%: Try to do the prediction separately and then use maths to put them back together.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# scale the data:
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

# apply the regression training function again.
r2_list, y_pred = regression_training(X_train_scaled, X_test_scaled, y_train, y_test, plot=True, output_y_pred=True)
y_pred_plus = y_pred

# do the same for y minus and collect the prediction
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.1, random_state=0)
# scale the data:
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

# apply the regression training function again.
r2_list, y_pred_minus = regression_training(X_train_scaled, X_test_scaled, y_train, y_test, plot=True, output_y_pred=True)

# compute the k1 and k2
logk1 = (y_pred_plus + y_pred_minus)/2
logk2 = (y_pred_plus - y_pred_minus)/2
