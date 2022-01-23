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
yplus = dfk_plus['Et_eV_2']

# ##
# we can see that the best behaviour is Random Forest: plot the graph mannually
X_train, X_test, y_train, y_test_plus = train_test_split(X, yplus, test_size=0.1)
# scale the data:
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
# do the training
r2, prediction_plus = regression_training(X_train_scaled, X_test_scaled, y_train, y_test_plus, plot=True, output_y_pred=True)

# do the same for Et minus:
dfk_minus = dfk[dfk['bandgap_1']==0]
# define X and y
X = dfk_minus.drop(['logk_1', 'logk_2', 'bandgap_1', 'bandgap_2', 'Et_eV_1', 'Et_eV_2'], axis=1)
X = np.log(X)
yminus = dfk_minus['Et_eV_2']
# we can see that the best behaviour is Random Forest: plot the graph mannually
X_train, X_test, y_train, y_test_minus = train_test_split(X, yminus, test_size=0.1)
# scale the data:
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
# do the training
r2, prediction_minus = regression_training(X_train_scaled, X_test_scaled, y_train, y_test_minus, plot=True, output_y_pred=True)

realE = np.concatenate((np.array(y_test_minus), np.array(y_test_plus)), axis=0)
predictedE = np.concatenate((prediction_minus['Random Forest'], prediction_plus['Support Vector']))
# plot the real vs predicted:
plt.figure()
plt.scatter(realE, predictedE)
plt.xlabel('Real Et (eV)')
plt.ylabel('Predicted Et (eV)')
plt.title('Real vs predicted for Et1 for single two level defect')
plt.show()
