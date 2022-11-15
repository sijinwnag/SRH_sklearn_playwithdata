# %%---import libraries:
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


# %%-- parameters definition
F1list = [0.881, 0.609, 0.772, 0.757, 0.839,  0.802, 0.782]
Modelname_list = ['Neural Network (0.881)', 'Naive Bayes (0.609)', 'Adaptive boosting (0.772)', 'Gradient Boosting (0.757)', 'Random Forest (0.839)', 'Suppor Vector Machine (0.802)', 'K Nearest Neighbor (0.782)']
erros = {'Neural Network (0.881)': 0.012, 'Naive Bayes (0.609)': 0.004, 'Adaptive boosting (0.772)': 0.003, 'Gradient Boosting (0.757)': 0.003, 'Random Forest (0.839)': 0.002, 'Suppor Vector Machine (0.802)': 0.002, 'K Nearest Neighbor (0.782)': 0.001}
erros2 = [0.012, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001]
# %%-

F1 = pd.DataFrame(np.transpose(F1list), index=Modelname_list)

# %%--bartchart
plt.figure(facecolor='white')
plt.barh(Modelname_list, F1list, error_kw=erros)
# plt.title('store inventory')
plt.xlabel('F-1 score', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.savefig('f1', bbox_inches='tight')
plt.show()
# %%-
