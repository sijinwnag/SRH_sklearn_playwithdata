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



class table_visualization():
    '''
    This class will visualize the R2 tables for n and p type material.
    '''

    def __init__(self):
        '''
        initialize by defining the table two panda dataframe.
        '''
        self.n_R2_table = pd.DataFrame([[0.933, 0.724, 0.845, 0.841, 0.737, 0.897],
        [0.761, 0.663, 0.835, 0.924, 0.908, 0.924],
        [0.68, 0.76, 0.696, 0.819, 0.731, 0.92],
        [0.61, 0.678, 0.726, 0.934, 0.935, 0.933]],
        columns=['$E_{t1}$', '$E_{t2}$', '$\sigma_{n1}$', '$\sigma_{p1}$', '$\sigma_{n2}$', '$\sigma_{p2}$'],
        index=['Set11', 'Set10', 'Set01', 'Set00'])

        self.p_R2_table = pd.DataFrame([[0.972, 0.919, 0.939, 0.643, 0.726, 0.726],
        [0.936, 0.987, 0.941, 0.821, 0.807, 0.703],
        [0.813, 0.916, 0.726, 0.693, 0.689, 0.783],
        [0.827, 0.882, 0.739, 0.934, 0.849, 0.766]],
        columns=['$E_{t1}$', '$E_{t2}$', '$\sigma_{n1}$', '$\sigma_{p1}$', '$\sigma_{n2}$', '$\sigma_{p2}$'],
        index=['Set11', 'Set10', 'Set01', 'Set00'])

        self.np_R2_table = pd.DataFrame([[0.98, 0.922, 0.9, 0.842, 0.74, 0.9],
        [0.936, 0.987, 0.941, 0.921, 0.907, 0.923],
        [0.811, 0.917, 0.726, 0.82, 0.789, 0.93],
        [0.829, 0.883, 0.74, 0.935, 0.936, 0.934]],
        columns=['$E_{t1}$', '$E_{t2}$', '$\sigma_{n1}$', '$\sigma_{p1}$', '$\sigma_{n2}$', '$\sigma_{p2}$'],
        index=['Set11', 'Set10', 'Set01', 'Set00'])


    def n_p_plot(self):
        '''
        Take the average of each column and plot a two column bar plot.
        '''
        # take the average along y axis
        n_av = self.n_R2_table.mean(axis=0)
        p_av = self.p_R2_table.mean(axis=0)
        av = pd.concat([n_av, p_av], axis=1)
        av.columns=['n type', 'p type']
        # print(av)
        # plot the figure.
        plt.figure(facecolor='white')
        av.plot(kind='bar')
        plt.title('Compare n-type and p-type')
        plt.ylabel('$R^2$' + ' score')
        plt.xticks(rotation=0)
        plt.ylim([0.6, 1])
        # export the image
        plt.savefig('npcompare.jpg')
        plt.show()


    def n_p_both_plot(self):
        '''
        Take the average of each column and plot a two column bar plot.
        '''
        # take the average along y axis
        n_av = self.n_R2_table.mean(axis=0)
        p_av = self.p_R2_table.mean(axis=0)
        np_av = self.np_R2_table.mean(axis=0)
        av = pd.concat([n_av, p_av, np_av], axis=1)
        av.columns=['n type', 'p type', 'n type and p type']
        # bars = ('$E_{t1}$', '$E_{t2}$', '', 'D', 'E')
        # print(av)
        # plot the figure.
        plt.figure(facecolor='white')
        av.plot(kind='bar')
        plt.title('Compare n, p and np together')
        plt.ylabel('$R^2$' + ' score')
        plt.xticks(rotation=0)
        plt.ylim([0.6, 1])
        # plt.legend(loc='upper right')
        # export the image
        plt.savefig('npcompare.jpg')
        plt.show()


    def set_plot(self, param='Et1'):
        '''
        Take the average of n and p table and plot the bartchart comparing R2
        '''
        # extract the column corresponding to the parameter.
        n_param = self.n_R2_table[param]
        p_param = self.p_R2_table[param]
        df = pd.concat([n_param, p_param], axis=1)
        # df = df.mean(axis=1)
        df.columns = ['n type', 'p type']

        # plot the figure.
        plt.figure(facecolor='white')
        df.plot(kind='bar')
        plt.title('Compare different sets for ' + '$E_{t2}$' + ' prediction')
        plt.ylabel('$R^2$' + ' score')
        plt.xticks(rotation=0)
        plt.ylim([0.6, 1])
        # export the image
        plt.legend(loc='upper right')
        plt.savefig('setscompare.jpg')
        plt.show()


# %%--
ob1 = table_visualization()
ob1.n_p_both_plot()
# ob1.set_plot(param='Et2')
# %%-
