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
################################################################################
# function definition

# define a function called 'outliers' which returns a list of index of outliers
# IQR = Q3-Q1
# boundary: +- 1.5*IQR
def outliers(df, ft, boundary):
    # input:
    # df: the data frame where we are looking for outliers
    # ft: the name of the feature that we are looking for outliers (string)
    # boundary: a number determine how wide is considered to be outliers, normally 1.5 or 3
    # output: a list of index of all the outliers.

    # start with calculating the quantiles
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)

    # calcualte the Interquartile range
    IQR = Q3 - Q1

    # define the upper and lower boundary for outliers.
    upper_bound = Q3 + boundary * IQR
    lower_bound = Q1 - boundary * IQR

    # collect the outliers
    out_list = df.index[(df[ft]<lower_bound) | (df[ft]>upper_bound)]

    return out_list


# define a function to find outliers for a list of futures
def outlier_ft_list(df, ft_list, inner_fence=True):
    # input:
    # df: the data frame where we are looking for outliers
    # ft: the name of the feature that we are looking for outliers (string)
    # inner_fence: if it is set to be true then the boundary is +-1.5*IQR, otherwise +-3*IQR
    # output: a list of index of all outliers for any feature in the ft_list.

    # decide whether use inner fence as outlier boundary or the outer fence.
    if inner_fence==True:
        boundary = 1.5
    else:
        boundary = 3

    out_list = []
    # find the outliers for each feature in a for loop
    for ft in ft_list:
        out_list.extend(outliers(df, ft, boundary))

    # remove the duplications
    out_list = list(dict.fromkeys(out_list))
    return out_list


def pre_processor(df):
    # input:
    # df: a dataframe of Walmart data csv.
    # output:
    # X_train_scaled
    # X-test_scaled.
    # y_train
    # y_test


    # identify extract the useful columns
    # Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp logk bandgap are all y
    # here we only required to find k or logk (we do not know them when doing regression)
    # since logk has less range, we peak log k instead of k
    # Therefore, delete: Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp bandgap
    delete_col = ['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap']
    dfk = df.drop(delete_col, axis=1)

    # train test split:
    X = np.log(X)
    X = dfk.drop(['logk'], axis=1)
    y = dfk['logk']
    # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # scale the data:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


################################################################################
# load the data.
df = pd.read_csv(r'C:\Users\sijin wang\Desktop\TOR_dell\literature_review\playing_with_data\lifetime_dataset_example.csv')
# repeat for 5 times: why it does not plot (using logX)
r2_frame_log = regression_repeat(df, 5)
print('finished')
r2_frame_log

# compare the r2 data of using logX with the one using X directly
################################################################################
# redefine the processor:
def pre_processor(df):
    # input:
    # df: a dataframe of Walmart data csv.
    # output:
    # X_train_scaled
    # X-test_scaled.
    # y_train
    # y_test


    # identify extract the useful columns
    # Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp logk bandgap are all y
    # here we only required to find k or logk (we do not know them when doing regression)
    # since logk has less range, we peak log k instead of k
    # Therefore, delete: Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp bandgap
    delete_col = ['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap']
    dfk = df.drop(delete_col, axis=1)

    # train test split:
    # X = np.log(X)
    X = dfk.drop(['logk'], axis=1)
    y = dfk['logk']
    # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # scale the data:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


###############################################################################
# redo the training with only X data:
# load the data.
# df = pd.read_csv(r'C:\Users\sijin wang\Desktop\TOR_dell\literature_review\playing_with_data\lifetime_dataset_example.csv')
# repeat for 5 times: why it does not plot (using logX)
r2_frame = regression_repeat(df, 5)
print('finished')
r2_frame
