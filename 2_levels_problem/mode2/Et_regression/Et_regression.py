# choose: Mode2
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
from function_for_trainings import regression_repeat, regression_training, classification_training, classification_repeat

################################################################################
# data pre processing:

# load the data.
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\2level_defects.csv')
# df.head()
# identify extract the useful columns
# Drop first column of dataframe
df = df.iloc[: , 1:]
delete_col = delete_col = ['Name', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2']
dfk = df.drop(delete_col, axis=1)

# extract thet ones that are two level defects.
dfk = dfk[dfk['Mode']=='Single two-level']
dfk = dfk.drop(['Mode'], axis=1)
# extract the defects that have bandgap_1 being 1:
dfk_plus = dfk[dfk['bandgap_1']==1]
# define X and y
X = dfk_plus.drop(['logk_1', 'logk_2', 'bandgap_1', 'bandgap_2', 'Et_eV_1', 'Et_eV_2'], axis=1)
X = np.log(X)
y = dfk_plus['Et_eV_1']

# do the regression.
r2_frame = regression_repeat(X, y, 1, plot=True)

# do the regresssion for the other half of bandgap.
dfk_minus = dfk[dfk['bandgap_1']==0]
# define X and y
X = dfk_minus.drop(['logk_1', 'logk_2', 'bandgap_1', 'bandgap_2', 'Et_eV_1', 'Et_eV_2'], axis=1)
X = np.log(X)
y = dfk_minus['Et_eV_1']

# do the regression.
r2_frame = regression_repeat(X, y, 1, plot=True)
