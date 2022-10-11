'''This file aims to plot state population vs prediction error '''


# %%-- Imports
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
import sys
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# uncomment the below line for dell laptop only
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
import scipy.constants as sc
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import sys
from MLobject_tlevel import *
file_path = r"C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\simulation_data\Etnonordered\p\set11\set11_80k.csv"
test_file_path = r"C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\simulation_data\Etnonordered\p\set11\set11_8k.csv"
# %%-


# %%-- Perform ML:
'''
This session will load the data, then add the prediction column at the end of each row.
'''

# load the training data:
pd_data = pd.read_csv(file_path)

# select the lifetime data:
select_X_list = []
for string in pd_data.columns.tolist():
    if string[0].isdigit():
        select_X_list.append(string)
pd_lifetime = pd_data[select_X_list] # take the lifetime as X, delete any column that does not start with a number.

# take the log of hte lifetime.
X_train = np.log10(pd_lifetime)

# define the scalor:
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# define the ML model.
model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
# traing the ML model.

# %%-
