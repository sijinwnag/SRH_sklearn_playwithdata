# %%-- imports
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
# %%-


real_Et1 = 0.15
real_Et2 = -0.3
predicted_Et1 = 0.145
predicted_Et2 = -0.39

# for the four value into pd dataframe.
data = pd.DataFrame([[real_Et1, real_Et2], [predicted_Et1, predicted_Et2]], columns=['Et1', 'Et2'], index=['Literature value', 'ML prediction'])
data = np.transpose(data)
# %%--
plt.figure()
data.plot(kind='bar')
plt.show()
# %%-
