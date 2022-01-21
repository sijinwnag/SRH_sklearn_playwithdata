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
delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap', 'logk']
dfk = df.drop(delete_col, axis=1)
# we also need to make sure to delete all the rows that has Et < 0:
dfk = dfk[dfk['Et_eV']<0]

# define X and y
X = dfk.drop(['Et_eV'], axis=1)
y = dfk['Et_eV']
# send it to regression repeat to train and evaluate the model.
r2_frame = regression_repeat(X, y, 5)

# instead of using x, try using logX
X = np.log(X)
r2_frame_log = regression_repeat(X, y, 5)

# compare the average r2 score for log and no log:
r2_av_log = np.average(r2_frame_log, axis=0)
r2_av = np.average(r2_frame, axis=0)
models_names = ['KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector']
df_plot = pd.DataFrame({'using original X': r2_av, 'using logX': r2_av_log}, index=models_names)
ax = df_plot.plot.barh()
ax.legend(bbox_to_anchor=(1.4, 0.55))
plt.title('the R2 scores for training using original X vs using logX')

# do a pair plot to visualize the data better
df_pairplot = df.drop(['Name', 'Sn_cm2', 'Sp_cm2', 'logSn', 'logSp', 'k'], axis=1)
df_pairplot = df_pairplot.iloc[: , :4]
df_pairplot.head()
df_pairplot = df_pairplot.rename(columns={'200K_5100000000000000.0cm-3_ 10000000000000.0cm-3': 'lifetime data'})
sn.pairplot(df_pairplot)

# do pairplot to compare the logX
df_pairplotlog = df_pairplot
df_pairplotlog.iloc[:, 3:] = np.log(df_pairplotlog.iloc[:, 3:])
sn.pairplot(df_pairplot)
