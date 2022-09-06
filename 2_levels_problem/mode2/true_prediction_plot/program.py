# %%-- Imports:
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors
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
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
import scipy.constants as sc
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
from sklearn.inspection import permutation_importance
import sympy as sym
# %%-


# %%-- define the object and plot it.
# define the path list.
path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_1set11_800k.csv'
path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_2set11_800k.csv'
path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_1set11_800k.csv'
path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_2set11_800k.csv'
path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_1set11_800k.csv'
path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_2set11_800k.csv'
pathlist = [path1, path2, path3, path4, path5, path6]
# define the R2.
R2list = [0.972, 0.645, 0.917, 0.732, 0.938, 0.727]
# define MAE:
MAElist = [0.013, 0.064, 0.128, 0.419, 0.151, 0.412]
# define title:
title1 = '$E_{t1}$' + '(eV)'
title2 = '$E_{t1}$' + '(eV)'
title3 = ''
# %%-
