# plot a histagram for Et for level 1 and level 2:
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

df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\2level_defects.csv')

Et1 = df['Et_eV_1']
Et2 = df['Et_eV_2']
# plot the histogram:
plt.figure()
plt.hist(Et1)
plt.ylabel('number of defect')
plt.xlabel('Et')
plt.title('Histogram for Et1')

plt.figure()
plt.hist(Et2)
plt.ylabel('number of defect')
plt.xlabel('Et')
plt.title('Histogram for Et2')
