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
# %%--set 11 p type:
# path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_1set11_800k.csv'
# path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\Et_eV_2set11_800k.csv'
# path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_1set11_800k.csv'
# path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSn_2set11_800k.csv'
# path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_1set11_800k.csv'
# path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\p\set11_800k\trial2\logSp_2set11_800k.csv'
# pathlist = [path1, path2, path3, path4, path5, path6]
# # define the R2.
# R2list = [0.972, 0.643, 0.919, 0.726, 0.939, 0.726]
# # define MAE:
# MAElist = [0.013, 0.064, 0.126, 0.422, 0.15, 0.412]
# title1 = '$E_{t1}$' + '(eV)'
# title2 = '$E_{t2}$' + '(eV)'
# title3 = 'log$(\sigma_{n1})$'
# title4 = 'log$(\sigma_{n2})$'
# title5 = 'log$(\sigma_{p1})$'
# title6 = 'log$(\sigma_{p2})$'
# titlelist = [title1, title2, title3, title4, title5, title6]
# filenamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-
# %%--set 11 n type:
path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_1set11_800k_n.csv'
path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\Et_eV_2set11_800k_n.csv'
path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_1set11_800k_n.csv'
path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSn_2set11_800k_n.csv'
path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_1set11_800k_n.csv'
path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logSp_2set11_800k_n.csv'
path7 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_1set11_800k_n.csv'
path8 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set11\n\set11_800k\set_11_n\logk_2set11_800k_n.csv'
pathlist = [path1, path2, path3, path4, path5, path6, path7, path8]
# define the R2.
R2list = [0.933, 0.841, 0.724, 0.737, 0.845, 0.897, 0.775, 0.809]
# define MAE:
MAElist = [0.024, 0.037, 0.399, 0.398, 0.264, 0.178, 0.549, 0.462]
title1 = '$E_{t1}$' + '(eV)'
title2 = '$E_{t2}$' + '(eV)'
title3 = 'log$(\sigma_{n1})$'
title4 = 'log$(\sigma_{n2})$'
title5 = 'log$(\sigma_{p1})$'
title6 = 'log$(\sigma_{p2})$'
title7 = 'log$(k_1)$'
title8 = 'log$(k_2)$'
titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
filenamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2', 'k1', 'k2']
# %%-
# %%--set 10 p type:
path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\Et_eV_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSn_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\Re__ML_finish_training\logSp_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path7 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_12022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
path8 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\p\logk_22022-10-28-03-08-13_advanced example - multi_level_L_datasetID_0.csv'
pathlist = [path1, path2, path3, path4, path5, path6, path7, path8]
# define the R2.
R2list = [0.923, 0.761, 0.907, 0.848, 0.909, 0.718, 0.91, 0.783]
# define MAE:
MAElist = [0.026, 0.054, 0.147, 0.292, 0.192, 0.411, 0.314, 0.545]
title1 = '$E_{t1}$' + '(eV)'
title2 = '$E_{t2}$' + '(eV)'
title3 = 'log$(\sigma_{n1})$'
title4 = 'log$(\sigma_{n2})$'
title5 = 'log$(\sigma_{p1})$'
title6 = 'log$(\sigma_{p2})$'
title7 = 'log$(k_1)$'
title8 = 'log$(k_2)$'
titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
filenamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2', 'k1', 'k2']
# %%-
# %%--set 10 n type:
# path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_1set10_800k_n.csv'
# path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\Et_eV_2set10_800k_n.csv'
# path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_1set10_800k_n.csv'
# path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSn_2set10_800k_n.csv'
# path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_1set10_800k_n.csv'
# path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set10\n\800k\Re__\logSp_2set10_800k_n.csv'
# pathlist = [path1, path2, path3, path4, path5, path6]
# # define the R2.
# R2list = [0.761, 0.926, 0.663, 0.908, 0.835, 0.924]
# # define MAE:
# MAElist = [0.053, 0.026, 0.467, 0.192, 0.309, 0.125]
# title1 = '$E_{t1}$' + '(eV)'
# title2 = '$E_{t2}$' + '(eV)'
# title3 = 'log$(\sigma_{n1})$'
# title4 = 'log$(\sigma_{n2})$'
# title5 = 'log$(\sigma_{p1})$'
# title6 = 'log$(\sigma_{p2})$'
# titlelist = [title1, title2, title3, title4, title5, title6]
# filenamelist = ['Et1', 'Et2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-
# %%--set 01 p type:
path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_1set01_800k_p.csv'
path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\Et_eV_2set01_800k_p.csv'
path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_1set01_800k_p.csv'
path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logk_2set01_800k_p.csv'
path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_1set01_800k_p.csv'
path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSn_2set01_800k_p.csv'
path7 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_1set01_800k_p.csv'
path8 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set01\p\800k\set01_p\set01_p\logSp_2set01_800k_p.csv'
pathlist = [path1, path2, path3, path4, path5, path6, path7, path8]
# define the R2.
R2list = [0.815, 0.694, 0.837, 0.753, 0.916, 0.684, 0.731, 0.782]
# define MAE:
MAElist = [0.038, 0.06, 0.415, 0.574, 0.158, 0.441, 0.391, 0.358]
# define title:
title1 = '$E_{t1}$' + '(eV)'
title2 = '$E_{t2}$' + '(eV)'
title3 = 'log$(k_1)$'
title4 = 'log$(k_2)$'
title5 = 'log$(\sigma_{n1})$'
title6 = 'log$(\sigma_{n2})$'
title7 = 'log$(\sigma_{p1})$'
title8 = 'log$(\sigma_{p2})$'
titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
filenamelist = ['Et1', 'Et2','k1', 'k2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-
# %%--set 01 n type:
# path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_1set00_800k.csv'
# path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_2set00_800k.csv'
# path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_1set00_800k.csv'
# path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_2set00_800k.csv'
# path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_1set00_800k.csv'
# path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_2set00_800k.csv'
# path7 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_1set00_800k.csv'
# path8 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_2set00_800k.csv'
# pathlist = [path1, path2, path3, path4, path5, path6, path7, path8]
# # define the R2.
# R2list = [0.831, 0.933, 0.802, 0.8, 0.883, 0.916, 0.848, 0.739, 0.767]
# # define MAE:
# MAElist = [0.039, 0.025, 0.471, 0.512, 0.198, 0.158, 0.262, 0.399, 0.358]
# # define title:
# title1 = '$E_{t1}$' + '(eV)'
# title2 = '$E_{t2}$' + '(eV)'
# title3 = 'log$(k_1)$'
# title4 = 'log$(k_2)$'
# title5 = 'log$(\sigma_{n1})$'
# title6 = 'log$(\sigma_{n2})$'
# title7 = 'log$(\sigma_{p1})$'
# title8 = 'log$(\sigma_{p2})$'
# titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
# filenamelist = ['Et1', 'Et2','k1', 'k2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-
# %%--set 00 p type:
# path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_1set00_800k.csv'
# path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\Et_eV_2set00_800k.csv'
# path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_1set00_800k.csv'
# path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logk_2set00_800k.csv'
# path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_1set00_800k.csv'
# path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSn_2set00_800k.csv'
# path7 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_1set00_800k.csv'
# path8 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\p\800k\set00_p\logSp_2set00_800k.csv'
# pathlist = [path1, path2, path3, path4, path5, path6, path7, path8]
# # define the R2.
# R2list = [0.831, 0.933, 0.802, 0.8, 0.883, 0.916, 0.848, 0.739, 0.767]
# # define MAE:
# MAElist = [0.039, 0.025, 0.471, 0.512, 0.198, 0.158, 0.262, 0.399, 0.358]
# # define title:
# title1 = '$E_{t1}$' + '(eV)'
# title2 = '$E_{t2}$' + '(eV)'
# title3 = 'log$(k_1)$'
# title4 = 'log$(k_2)$'
# title5 = 'log$(\sigma_{n1})$'
# title6 = 'log$(\sigma_{n2})$'
# title7 = 'log$(\sigma_{p1})$'
# title8 = 'log$(\sigma_{p2})$'
# titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
# filenamelist = ['Et1', 'Et2','k1', 'k2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-
# %%--set 00 n type:
# path1 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_1set00_n_800k.csv'
# path2 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\Et_eV_2set00_n_800k.csv'
# path3 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_1set00_n_800k.csv'
# path4 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logk_2set00_n_800k.csv'
# path5 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_1set00_n_800k.csv'
# path6 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSn_2set00_n_800k.csv'
# path7 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_1set00_n_800k.csv'
# path8 = r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\withoutEt1larger\set00\n\800k\Re__\logSp_2set00_n_800k.csv'
# pathlist = [path1, path2, path3, path4, path5, path6, path7, path8]
# # define the R2.
# R2list = [0.609, 0.97, 0.7, 0.932, 0.68, 0.936, 0.727, 0.932]
# MAElist = [0.069, 0.014, 0.649, 0.256, 0.459, 0.154, 0.426, 0.108]
# # define title:
# title1 = '$E_{t1}$' + '(eV)'
# title2 = '$E_{t2}$' + '(eV)'
# title3 = 'log$(k_1)$'
# title4 = 'log$(k_2)$'
# title5 = 'log$(\sigma_{n1})$'
# title6 = 'log$(\sigma_{n2})$'
# title7 = 'log$(\sigma_{p1})$'
# title8 = 'log$(\sigma_{p2})$'
# titlelist = [title1, title2, title3, title4, title5, title6, title7, title8]
# filenamelist = ['Et1', 'Et2','k1', 'k2', 'Sn1', 'Sn2', 'Sp1', 'Sp2']
# %%-
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
    plt.scatter(true, prediction, label=('$R^2$' + '=' + str(R2list[k])) + ('; MAE' + '=' + str(MAElist[k])), alpha=alpha, color='green')
    plt.plot(true, true, color='r')
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    ax.set_aspect("equal")
    # plt.text(0, 0.5, alphabet[k], fontsize=20)
    plt.title(titlelist[k], fontsize=25)
    plt.legend(loc=4, framealpha=0.1, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if k>1 and k<6:
        plt.xticks(range(-17, -12))
        plt.yticks(range(-17, -12))
    plt.savefig(fname=str(filenamelist[k]) + '.png')
    plt.show()
# %%-
