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



# %%--Plot the preperation time
plt.figure(facecolor = 'white')
plt.bar(x = ['pure residual', 'pure ML'], height = [3200, 1300])
plt.ylabel('Computational time (min)')
plt.title('Computation time when ML is training')
plt.savefig('Time_compare.png')
plt.show()
# %%-

# %%-- Plot the predict time.
plt.figure(facecolor = 'white')
plt.bar(x = ['pure residual', 'pure ML'], height = [3200, 0.1/60])
plt.ylabel('Computational time (min)')
plt.title('Computation time after ML finish training')
plt.savefig('Time_compare.png')
plt.show()
# %%-
