"""
This file will demonstrate the R2 score vs sample size curve
The expected shape is to R2 score increases with sample size curve but the rate of increase will get slower as sample size further increase.

plan:
1. create a list of sample size fraction like [0.1, 0.2, 0.3, ...1]
2. load the whole dataframe and do the data pre processing for k regresion problem
3. for each fraction: randomly select the subset according to the fraction, calcualte the R2 score. and collect the R2 and sample sizes values.
4. plot the sample size vs R2 scores

"""
################################################################################
# import the libraries
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
#################################################################################
# load the data.
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# data pre processing:
# identify extract the useful columns
delete_col = ['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap']
dfk = df.drop(delete_col, axis=1)

# defien the fraction list:
fraction_list = np.arange(0.1, 1.1, 0.1)

# prepare an empty list to collect all the r2 scores and sample sizes
r2_list = []
samplesize_list = []
for fraction in fraction_list:
    # randomly select the subset of original data.
    dfk_subset = dfk.sample(frac=fraction)
    samplesize_list.append(np.shape(dfk_subset)[0])
    # define X and y
    X = dfk_subset.drop(['logk'], axis=1)
    y = dfk_subset['logk']

    # train the data and get a set of r2 scores
    r2_frame = regression_repeat(X, y, 5)
    # average the scores
    r2_frame = np.average(r2_frame, axis=0)
    # select the highst scores
    r2_list.append(np.max(r2_frame))

# plot sample sizes vs best r2 scores
plt.figure()
plt.plot(samplesize_list, r2_list)
plt.xlabel('sample size')
plt.ylabel('average R2 scores')
plt.title('Sample size vs R2 scores for 5 itaratinos')
plt.show()
