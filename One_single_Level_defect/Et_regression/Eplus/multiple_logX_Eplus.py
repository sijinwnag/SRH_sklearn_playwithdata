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
sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata')
from function_for_trainings import regression_repeat, regression_training

################################################################################
# data pre processing:
# load the data.
# df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')
df = pd.read_csv(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\lifetime_dataset_example.csv')
# train and evaluate the models.
delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap', 'logk']
dfk = df.drop(delete_col, axis=1)
# we also need to make sure to delete all the rows that has Et < 0:
dfk = dfk[dfk['Et_eV']>0]

# define X and y
X = dfk.drop(['Et_eV'], axis=1)
y = dfk['Et_eV']
################################################################################
# machine learning.
r2scores = regression_repeat(X, y, 1)
# r2scores.to_csv('Etminus_diffmodels.csv')
# use r2scores to plot a barchart of average score for each model.
avr2scores = np.average(r2scores, axis=0)
# avr2scores
# create a barchart
plt.figure()
models = ('KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector')
plt.barh(models, avr2scores)
plt.ylabel('R2 score')
plt.title(' average R2 score for Et regression above intrinsic fermi energy')
plt.show()


################################################################################
# redefine the pro processor: this time we train and test with logX instead of X.
Xlog = np.log(X)

# train and evaluate the models.
r2scoreslog = regression_repeat(Xlog, y, 1)
# r2scores.to_csv('Etminus_diffmodels.csv')
# use r2scores to plot a barchart of average score for each model.
avr2scoreslog = np.average(r2scoreslog, axis=0)
# avr2scores
# create a barchart
plt.figure()
models = ('KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector')
plt.barh(models, avr2scoreslog)
plt.ylabel('R2 score')
plt.title(' average R2 score for Et regression above intrinsic fermi energy')
plt.show()
