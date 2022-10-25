# %%--
'''
n-type:
Try with 8k data: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.197 (true value is 0.15).

p-type:
Try with 8k data: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.145 (true value is 0.15).
Try with 80k data: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.145 (true value is 0.15).
'''
# %%-

# %%--
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
import math
# import the function file from another folder:
# use this line if on hp laptop:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\DPML')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Savedir_example')
from MLobject_tlevel import *
# %%-

# %%--Et1

# load the BO example.
BO_data = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\BO_validation\BO_ptype\2022-10-25-11-14-51_advanced example - multi_level_L_datasetID_0.csv')
# load the trianing data.
training_data = pd.read_csv(r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\simulation_data\Etnonordered\p\set10\2022_08_29_8k\2022-08-29-11-51-36_advanced example - multi_level_L_datasetID_0.csv')
# training_data.head()

# extract the lifetime.
BO_lifetime = BO_data.iloc[:,17:-2]
training_lifetime = training_data.iloc[:, 17:-2]
# BO_lifetime.head()
# training_lifetime.head()


# take log10
BO_lifetime_log = BO_lifetime.applymap(math.log10)
training_lifetime_log = training_lifetime.applymap(math.log10)


# go through scaler.
scaler = MinMaxScaler()
training_scaled = scaler.fit_transform(training_lifetime_log)
BO_scaled = scaler.transform(BO_lifetime_log)


# define the target variable.
y_train = training_data['Et_eV_1']
y_test = BO_data['Et_eV_1']
# y_train
# y_test

# define the model.
model = RandomForestRegressor(n_estimators=100, verbose=20)
# train the model.
model.fit(training_scaled, y_train)
# predict
print(model.predict(BO_scaled))
sys.stdout = open(r"Bo_validation_Et1.txt", "w")
sys.stdout.close()
# %%- Et_eV_1

# %%--Et2
sys.stdout = open(r"Bo_validation_Et1.txt", "w")
# load the BO example.
BO_data = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\BO_validation.csv')
# load the trianing data.
training_data = pd.read_csv(r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\simulation_data\Etnonordered\p\set10\2022_08_29_8k\2022-08-29-11-51-36_advanced example - multi_level_L_datasetID_0.csv')

# extract the lifetime.
BO_lifetime = BO_data.iloc[:,17:-2]
training_lifetime = training_data.iloc[:, 17:-2]

# take log10
BO_lifetime_log = np.log10(BO_lifetime)
training_lifetime_log = np.log10(training_lifetime)

# go through scaler.
scaler = MinMaxScaler()
training_scaled = scaler.fit_transform(training_lifetime_log)
BO_scaled = scaler.transform(BO_lifetime_log)

# define the target variable.
y_train = training_data['Et_eV_1']
y_test = BO_data['Et_eV_1']

# define the model.
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1)
# train the model.
model.fit(training_scaled, y_train)
# predict
print(model.predict(BO_scaled))
sys.stdout.close()
# %%- Et_eV_2
