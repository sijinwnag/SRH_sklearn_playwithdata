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
title2 = '$E_{t2}$' + '(eV)'
title3 = '$log(\sigma_{n1})$'
title4 = '$log(\sigma_{n2})$'
title5 = '$log(\sigma_{p1})$'
title6 = '$log(\sigma_{p2})$'
titlelist = [title1, title2, title3, title4, title5, title6]
filenamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-


# %%-- extract the data.
Truelist = []
predictionlist = []
for path in pathlist:
    data = pd.read_csv(path)
    # the second column is true value:
    true = np.array(data)[:, 1]
    Truelist.append(true)
    # the third column is the prediction value:
    prediction = np.array(data)[:, 2]
    predictionlist.append(prediction)
# Truelist and predictionlist are lists of numpy array.
# %%-


# %%-- calculate the transparency:
def transparency_calculator(datasize):
    '''
    This function will calcualte a suitable data transparency given the datasize for a scatter plot.

    input: datasize: an integer.
    '''
    # load the standardsize from the object
    standardsize=800
    if datasize>standardsize:
        alpha = standardsize/datasize*0.5
    else:
        alpha = 0.5
    return alpha

alpha = transparency_calculator(80000)
print('transparency is ' + str(alpha))
# %%-


# %%-- do the subplot:
fig, axs = plt.subplots(2, 3)
# define the index and the position of each subplot.
plot_position1 = [0, 0, 0, 1, 1, 1]
plot_position2 = [0, 1, 2, 0, 1, 2]
for k in range(len(Truelist)):
    # extract the values:
    truevalue = Truelist[k]
    prediction = predictionlist[k]
    axs[plot_position1[k], plot_position2[k]].scatter(truevalue, prediction, alpha=alpha)
    axs[plot_position1[k], plot_position2[k]].set_title(titlelist[k])
# %%-


# %%-- do the plots one by ones
alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for k in range(len(Truelist)):
    # plot the plot.
    fig= plt.figure(facecolor='white', figsize=(6, 6))
    ax = fig.add_subplot(111)
    true = Truelist[k]
    prediction = predictionlist[k]
    plt.scatter(true, prediction, label=('$R^2$' + '=' + str(R2list[k])) + ('  MAE' + '=' + str(MAElist[k])), alpha=alpha, color='green')
    plt.plot(true, true, color='r')
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    ax.set_aspect("equal")
    # plt.text(0, 0.5, alphabet[k], fontsize=20)
    plt.title(titlelist[k], fontsize=25)
    plt.legend(loc=4, framealpha=0.1, fontsize=20)
    plt.savefig(fname=str(filenamelist[k]) + '.png')
    plt.show()
# %%-
