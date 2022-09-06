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


class True_vs_prediction_plot:

    """
    Takes a 2D dataframe and plot the real vs prediction plot.
    """

    def __init__(self, path, R2, MAE, title):
        '''
        initialize the object by defining the path of the data, and load the data into pd dataframe from the given path.
        '''
        self.path = path
        self.data = pd.read_csv(path)
        self.standardsize=800
        # input the R2 score and MAE into the object.
        self.R2 = R2
        self.MAE = MAE
        self.title = title


    def transparency_calculator(self, datasize):
        '''
        This function will calcualte a suitable data transparency given the datasize for a scatter plot.

        input: datasize: an integer.
        '''
        # load the standardsize from the object
        standardsize=self.standardsize
        if datasize>standardsize:
            alpha = standardsize/datasize*0.5
        else:
            alpha = 0.5
        return alpha


    def plot(self):
        '''
        plot the real vs prediction plot
        '''
        # extract the true value and prediction value from the object.
        true = np.array(self.data)[:, 1]
        prediction = np.array(self.data)[:, 2]

        # plot the plot.
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        # calculate the transparency:
        alpha=self.transparency_calculator(len(true))
        print('transparency of scattering plot is ' + str(alpha))
        plt.scatter(true, prediction, label=('$R^2$' + '=' + str(self.R2)) + ('  MAE' + '=' + str(self.MAE)), alpha=alpha)
        plt.plot(true, true, color='r')
        plt.xlabel('True')
        plt.ylabel('Prediction')
        ax.set_aspect("equal")
        plt.title(self.title)
        plt.legend(loc=3, framealpha=0.1)
        # plt.savefig(str(self.singletask) + '.png')
        plt.show()


    def subplot(self):
        '''
        plot the true ve prediction into 6 different subplots:
        '''
        
