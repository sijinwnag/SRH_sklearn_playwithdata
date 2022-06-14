# %%---import libraries:
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
# import the function file from another folder:
# use this line if on hp laptop:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
from MLobject_tlevel import *
# %%-


class Dynamic_regression:

    """
    MyMLdata is an object that does the multi_step dynamic data generatino regression.
    """

    def datatraining(trainingset_path, validationset_path, repeat, parameter):
        '''
        This function input two dataset: one is training, the other is validation, and output the perdiction from training, the models and the evaluation results.

        input:
        trainingset_path, validationset_path: the paths to load the training and validation set data.
        repeat: the repeat number when training using hte training set.
        parameter: the parameters to predict during the training and validation.

        output:
        model_list: a list of trained model corresponding to each parameter.
        predict_list: the prediction for each maching learning task. the dimension of this array is: [datasize in validation list]*[different tasks]
        '''

        # define the maching learning object for step training.
        training_step1 = MyMLdata_2level(trainingset_path, 'bandgap1', repeat)
        # see if the function can return the model correctly for predicting first step.
        step1_parameter = parameter
        # prepare an empty list to collect model for each task:
        model_list = []
        # pr4epare an empty list to collect prediction for each task:
        predict_list = []
        # prepare an empty list to collect the evaluation score for each task:
        r2_list = []
        meanabs_list = []
        # defien the set we want to do validation on
        prediction_step1 = MyMLdata_2level(validationset_path, 'bandgap1',repeat)
        # iterate for each parameter
        for parameter in step1_parameter:
            # print(parameter)
            # defein the y to be trained using machine learning.
            training_step1.singletask = parameter
            # try to make it return the best R2 score model for all trials all models.
            r2_frame, y_prediction_frame, y_test_frame, selected_model, scaler = training_step1.regression_repeat(output_y_pred=True)
            # sanity check: see if it can select the best model based on the average R2 or mean absolute error.
            # print(selected_model)
            model_list.append(selected_model)
            # extract the data using pre-processor: X is the log of lifetime data, y is the colume we want to predict.
            prediction_step1.singletask = parameter
            X_test, y_test = prediction_step1.pre_processor()
            # scale the data, the data which the model is trained and validated should be the same scalter.
            X_scaled = scaler.fit_transform(X_test)
            # now do the prediction.
            y_predict = selected_model.predict(X_scaled) # the dimension of this array is: [datasize in validation list]*[different tasks]
            predict_list.append(y_predict)
            # evaluate the model using both R2 score and mean absolute error.
            r2 = r2_score(y_test, y_predict) # this is a float.
            r2_list.append(r2)
            meanabs = mean_absolute_error(y_test, y_predict)
            meanabs_list.append(meanabs)

        return model_list, predict_list, r2_list, meanabs_list
