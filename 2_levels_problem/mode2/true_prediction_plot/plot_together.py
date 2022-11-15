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


# %%-- inputs:
# Set 11 p type.
path_Et1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_1set11_800k.csv'
path_Et2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_2set11_800k.csv'
path_Sn1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_1set11_800k.csv'
path_Sn2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_2set11_800k.csv'
path_Sp1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_1set11_800k.csv'
path_Sp2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_2set11_800k.csv'
path_k1_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\logk_1set11_800k.csv'
path_k2_11_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\logk_2set11_800k.csv'
# Set 11 n type.
path_Et1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_1set11_800k_n.csv'
path_Et2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_2set11_800k_n.csv'
path_Sn1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_1set11_800k_n.csv'
path_Sn2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_2set11_800k_n.csv'
path_Sp1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_1set11_800k_n.csv'
path_Sp2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_2set11_800k_n.csv'
path_k1_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_1set11_800k_n.csv'
path_k2_11_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_2set11_800k_n.csv'
# Set 10 p type.
path_Et1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Et2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sn1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sn2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sp1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_Sp2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_k1_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path_k2_10_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
# Set 10 n type.
path_Et1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_1set10_800k_n.csv'
path_Et2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_2set10_800k_n.csv'
path_Sn1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_1set10_800k_n.csv'
path_Sn2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_2set10_800k_n.csv'
path_Sp1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_1set10_800k_n.csv'
path_Sp2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_2set10_800k_n.csv'
path_k1_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\logk_1set10_800k_n.csv'
path_k2_10_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\logk_2set10_800k_n.csv'
# Set 01 p type.
path_Et1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_1set01_800k_p.csv'
path_Et2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_2set01_800k_p.csv'
path_k1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_1set01_800k_p.csv'
path_k2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_2set01_800k_p.csv'
path_Sn1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_1set01_800k_p.csv'
path_Sn2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_2set01_800k_p.csv'
path_Sp1_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_1set01_800k_p.csv'
path_Sp2_01_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_2set01_800k_p.csv'
# Set 01 n type.
path_Et1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\Et_eV_1set01_800k_n.csv'
path_Et2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\Et_eV_2set01_800k_n.csv'
path_Sn1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSn_1set01_800k_n.csv'
path_Sn2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSn_2set01_800k_n.csv'
path_Sp1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSp_1set01_800k_n.csv'
path_Sp2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\set01_n_800k\logSp_2set01_800k_n.csv'
path_k1_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\logk_1set01_800k_n.csv'
path_k2_01_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\n\logk_2set01_800k_n.csv'
# Set 00 p type.
path_Et1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_1set00_800k.csv'
path_Et2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_2set00_800k.csv'
path_k1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_1set00_800k.csv'
path_k2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_2set00_800k.csv'
path_Sn1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_1set00_800k.csv'
path_Sn2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_2set00_800k.csv'
path_Sp1_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_1set00_800k.csv'
path_Sp2_00_p = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_2set00_800k.csv'
# Set 00 n type.
path_Et1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_1set00_n_800k.csv'
path_Et2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_2set00_n_800k.csv'
path_k1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_1set00_n_800k.csv'
path_k2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_2set00_n_800k.csv'
path_Sn1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_1set00_n_800k.csv'
path_Sn2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_2set00_n_800k.csv'
path_Sp1_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_1set00_n_800k.csv'
path_Sp2_00_n = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_2set00_n_800k.csv'
# put everything in a list:
# both n and p.
# Et1list = [path_Et1_11_p, path_Et1_11_n, path_Et1_10_p, path_Et1_10_n, path_Et1_01_p, path_Et1_01_n, path_Et1_00_p, path_Et1_00_n]
# Et2list = [path_Et2_11_p, path_Et2_11_n, path_Et2_10_p, path_Et2_10_n, path_Et2_01_p, path_Et2_01_n, path_Et2_00_p, path_Et2_00_n]
# Sn1list = [path_Sn1_11_p, path_Sn1_11_n, path_Sn1_10_p, path_Sn1_10_n, path_Sn1_01_p, path_Sn1_01_n, path_Sn1_00_p, path_Sn1_00_n]
# Sn2list = [path_Sn2_11_p, path_Sn2_11_n, path_Sn2_10_p, path_Sn2_10_n, path_Sn2_01_p, path_Sn2_01_n, path_Sn2_00_p, path_Sn2_00_n]
# Sp1list = [path_Sp1_11_p, path_Sp1_11_n, path_Sp1_10_p, path_Sp1_10_n, path_Sp1_01_p, path_Sp1_01_n, path_Sp1_00_p, path_Sp1_00_n]
# Sp2list = [path_Sp2_11_p, path_Sp2_11_n, path_Sp2_10_p, path_Sp2_10_n, path_Sp2_01_p, path_Sp2_01_n, path_Sp2_00_p, path_Sp2_00_n]
# p.
Et1list = [path_Et1_11_p, path_Et1_10_p, path_Et1_01_p, path_Et1_00_p]
Et2list = [path_Et2_11_p, path_Et2_10_p, path_Et2_01_p, path_Et2_00_p]
Sn1list = [path_Sn1_11_p,  path_Sn1_10_p,path_Sn1_01_p, path_Sn1_00_p]
Sn2list = [path_Sn2_11_p, path_Sn2_10_p, path_Sn2_01_p, path_Sn2_00_p]
Sp1list = [path_Sp1_11_p, path_Sp1_10_p, path_Sp1_01_p, path_Sp1_00_p]
Sp2list = [path_Sp2_11_p, path_Sp2_10_p, path_Sp2_01_p, path_Sp2_00_p]
k1list = [path_k1_11_p, path_k1_10_p, path_k1_01_p, path_k1_00_p]
k2list = [path_k2_11_p, path_k2_10_p, path_k2_01_p, path_k2_00_p]
# n
# Et1list = [path_Et1_11_n, path_Et1_10_n, path_Et1_01_n, path_Et1_00_n]
# Et2list = [path_Et2_11_n, path_Et2_10_n, path_Et2_01_n, path_Et2_00_n]
# Sn1list = [path_Sn1_11_n,  path_Sn1_10_n,path_Sn1_01_n, path_Sn1_00_n]
# Sn2list = [path_Sn2_11_n, path_Sn2_10_n, path_Sn2_01_n, path_Sn2_00_n]
# Sp1list = [path_Sp1_11_n, path_Sp1_10_n, path_Sp1_01_n, path_Sp1_00_n]
# Sp2list = [path_Sp2_11_n, path_Sp2_10_n, path_Sp2_01_n, path_Sp2_00_n]
# k1list = [path_k1_11_n, path_k1_10_n, path_k1_01_n, path_k1_00_n]
# k2list = [path_k2_11_n, path_k2_10_n, path_k2_01_n, path_k2_00_n]
# %%-


# %%-- Extract the data.
Truelist = []
predictionlist = []
for path in Et1list:
    data = pd.read_csv(path)
    # the second column is true value:
    true = np.array(data)[:, 1]
    Truelist.append(true)
    # the third column is the prediction value:
    prediction = np.array(data)[:, 2]
    predictionlist.append(prediction)

# flattern the list into 1d array.
Truelist = np.array(Truelist).flatten()
predictionlist = np.array(predictionlist).flatten()

# subsampling.
sampleindex = np.random.randint(0, np.shape(Truelist)[0], 10000)
Truelist = Truelist[sampleindex]
predictionlist = predictionlist[sampleindex]
error = np.absolute(Truelist-predictionlist)

# calculate evaluation matrix.
R2 = r2_score(Truelist, predictionlist)
print(R2)
MAE = mean_absolute_error(Truelist, predictionlist)
print(MAE)
# %%-


# %%-- Plotting
fig= plt.figure(facecolor='white', figsize=(6, 6))
ax = fig.add_subplot(111)
true = Truelist
prediction = predictionlist
plt.scatter(true, prediction, label=('$R^2$' + '=' + str(round(R2, 3))) + ('; MAE' + '=' + str(round(MAE, 3))), alpha=0, color='green')
plt.plot(true, true, color='r')
plt.xlabel('True', fontsize=20)
plt.ylabel('Prediction', fontsize=20)
ax.set_aspect("equal")
# plt.text(0, 0.5, alphabet[k], fontsize=20)
# plt.title('$E_{t1}$', fontsize=25)
plt.title('True vs prediction plot', fontsize=20)
# plt.legend(loc=4, framealpha=0.1, fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(fname=str('Et1') + '.png', bbox_inches='tight')
plt.show()
# %%-


# %%-- Plot all parameters together
filetnamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2', 'k1', 'k2']
title1 = '$E_{t1}$' + ' (eV)'
title2 = '$E_{t2}$' + ' (eV)'
title3 = 'log$(\sigma_{n1})$'
title4 = 'log$(\sigma_{n2})$'
title5 = 'log$(\sigma_{p1})$'
title6 = 'log$(\sigma_{p2})$'
title7 = 'log$(k_1)$'
title8 = 'log$(k_2)$'
titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
counter = 0
for task in [Et1list, Et2list, Sn1list, Sn2list, Sp1list, Sp2list, k1list, k2list]:
    # extract the dataset.
    filename = filetnamelist[counter]
    Truelist = []
    predictionlist = []
    for path in task:
        data = pd.read_csv(path)
        # the second column is true value:
        true = np.array(data)[:, 1]
        Truelist.append(true)
        # the third column is the prediction value:
        prediction = np.array(data)[:, 2]
        predictionlist.append(prediction)

    # flattern the list into 1d array.
    Truelist = np.array(Truelist).flatten()
    predictionlist = np.array(predictionlist).flatten()

    # subsampling.
    sampleindex = np.random.randint(0, np.shape(Truelist)[0], 10000)
    Truelist = Truelist[sampleindex]
    predictionlist = predictionlist[sampleindex]

    # calculate evaluation matrix.
    R2 = r2_score(Truelist, predictionlist)
    print(R2)
    MAE = mean_absolute_error(Truelist, predictionlist)
    print(MAE)

    # plotting without centre line.
    fig= plt.figure(facecolor='white', figsize=(6, 6))
    ax = fig.add_subplot(111)
    true = Truelist
    prediction = predictionlist
    plt.scatter(true, prediction, label=('$R^2$' + '=' + str(round(R2, 3))) + ('; $MAE$' + '=' + str(round(MAE, 3))), alpha=0.01, color='green')
    # plt.plot(true, true, color='r')
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    ax.set_aspect("equal")
    # plt.text(0, 0.5, alphabet[k], fontsize=20)
    plt.title(str(titlelist[counter]), fontsize=25)
    plt.legend(loc=4, framealpha=0.1, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if filename[0] == 'S':
        plt.xticks(range(-17, -12))
        plt.yticks(range(-17, -12))
    plt.savefig(fname=str(filetnamelist[counter]) + 'without center line' + '.png', bbox_inches='tight')
    plt.show()

    # plotting with centre line.
    fig= plt.figure(facecolor='white', figsize=(6, 6))
    ax = fig.add_subplot(111)
    true = Truelist
    prediction = predictionlist
    plt.scatter(true, prediction, label=('$R^2$' + '=' + str(round(R2, 3))) + ('; $MAE$' + '=' + str(round(MAE, 3))), alpha=0.01, color='green')
    plt.plot(true, true, color='r')
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    ax.set_aspect("equal")
    # plt.text(0, 0.5, alphabet[k], fontsize=20)
    plt.title(str(titlelist[counter]), fontsize=25)
    plt.legend(loc=4, framealpha=0.1, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if filename[0] == 'S':
        # print(filename)
        plt.xticks(range(-17, -12))
        plt.yticks(range(-17, -12))
    plt.savefig(fname=str(filetnamelist[counter]) + 'with center line' + '.png', bbox_inches='tight')
    plt.show()

    counter = counter + 1
# %%-


# %%-- Plot itself vs error.
filetnamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
title1 = '$E_{t1}$' + '(eV)'
title2 = '$E_{t2}$' + '(eV)'
title3 = 'log$(\sigma_{n1})$'
title4 = 'log$(\sigma_{n2})$'
title5 = 'log$(\sigma_{p1})$'
title6 = 'log$(\sigma_{p2})$'
title7 = 'log$(k_1)$'
title8 = 'log$(k_2)$'
titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
counter = 0
for task in [Et1list, Et2list, Sn1list, Sn2list, Sp1list, Sp2list]:
    # extract the dataset.
    Truelist = []
    predictionlist = []
    for path in task:
        data = pd.read_csv(path)
        # the second column is true value:
        true = np.array(data)[:, 1]
        Truelist.append(true)
        # the third column is the prediction value:
        prediction = np.array(data)[:, 2]
        predictionlist.append(prediction)

    # flattern the list into 1d array.
    Truelist = np.array(Truelist).flatten()
    predictionlist = np.array(predictionlist).flatten()

    # subsampling.
    sampleindex = np.random.randint(0, np.shape(Truelist)[0], 10000)
    Truelist = Truelist[sampleindex]
    predictionlist = predictionlist[sampleindex]
    error = np.absolute(Truelist - predictionlist)


    # plotting without centre line.
    fig= plt.figure(facecolor='white', figsize=(6, 6))
    ax = fig.add_subplot(111)
    true = Truelist[error>0.1]
    error = error[error>0.1]
    plt.scatter(true, error, alpha=0.1, color='green')
    # plt.plot(true, true, color='r')
    plt.xlabel('Value', fontsize=20)
    plt.ylabel('Prediction error', fontsize=20)
    ax.set_aspect("equal")
    # plt.text(0, 0.5, alphabet[k], fontsize=20)
    plt.title(str(titlelist[counter]), fontsize=25)
    # plt.legend(loc=4, framealpha=0.1, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(fname=str(filetnamelist[counter]) + 'error' + '.png', bbox_inches='tight')
    plt.show()


    counter = counter + 1
# %%-
