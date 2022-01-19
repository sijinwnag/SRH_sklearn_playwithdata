"""
This file do k regression for multiple times and take Log X befroe scaling the X.
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
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata')
from function_for_trainings import regression_repeat, regression_training

################################################################################
# data pre processing:

# load the data.
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# identify extract the useful columns
# Name	Et_eV	Sn_cm2	Sp_cm2	k	logSn	logSp	logk	bandgap are all y
# here we only required to find k or logk (we do not know them when doing regression)
# since logk has less range, we peak log k instead of k
# Therefore, delete: Name	Et_eV	Sn_cm2	Sp_cm2	k	logSn	logSp bandgap
delete_col = ['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap']
dfk = df.drop(delete_col, axis=1)

# define X and y
X = dfk.drop(['logk'], axis=1)
y = dfk['logk']

# send it to regression repeat to train and evaluate the model.
r2_frame = regression_repeat(X, y, 1)

# instead of using x, try using logX
X = np.log(X)
r2_frame_log = regression_repeat(X, y, 1)
