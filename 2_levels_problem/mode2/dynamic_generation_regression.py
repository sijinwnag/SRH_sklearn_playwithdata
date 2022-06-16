# %%-- to do:
'''
integrate the dynamic generation method into one object.
lets first make sure that the ML can do only 2 steps so we do not need to code 3 layers of for loop.
'''
# %%-


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
import sys
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
# from Si import *

# use this line for dell laptop:
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new')
# use this line for workstation:
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new\DPML')
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
from email.message import EmailMessage
import os
# %%-


class Dynamic_regression:

    """
    MyMLdata is an object that does the multi_step dynamic data generatino regression.
    """
    def __init__(self):
            # define the default parameter for data simulation.


            self.task = [['Et_eV_1', 'logSn_1', 'logSp_1'], ['Et_eV_2', 'logSn_2', 'logSp_2']]
            self.first_step_training_path = r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new\Savedir_example\outputs\small_dataset.csv'
            self.validation_path = r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new\Savedir_example\outputs\dummy_validation_11.csv'
            self.n_repeat = 2
            self.validationdata = pd.read_csv(self.validation_path)
            self.simulate_size = 80


    def datatraining(self, trainingset_path, repeat, parameter):
        '''
        This function input two dataset: one is training, the other is validation, and output the perdiction from training, the models and the evaluation results.

        input:
        trainingset_path, validationset_path: the paths to load the training and validation set data.
        repeat: the repeat number when training using hte training set.
        parameter: the parameters to predict during the training and validation.

        output:
        model_list: a list of trained model corresponding to each parameter.
        scaler_list: a list of scaler corresponding to each model, because the model is trained based on scale(log(X)) and Y
        y_predictions: the prediction for each task for each dataset.

        for the flow chart of this function, see the file: dynamic_regressor.ppt
        '''
        # print(self.task)
        # define the maching learning object for step training.
        training_step1 = MyMLdata_2level(trainingset_path, 'bandgap1', repeat)
        # see if the function can return the model correctly for predicting first step.
        step1_parameter = parameter
        # prepare an empty list to collect model for each task:
        model_list = []
        scaler_list = []
        y_predictions = []
        # iterate for each parameter
        for parameter in step1_parameter:
            print('training ' + str(parameter))
            # defein the y to be trained using machine learning.
            training_step1.singletask = parameter
            # try to make it return the best R2 score model for all trials all models.
            r2_frame, y_prediction_frame, y_test_frame, selected_model, scaler = training_step1.regression_repeat(output_y_pred=True)
            # sanity check: see if it can select the best model based on the average R2 or mean absolute error.
            # print(selected_model)
            model_list.append(selected_model)
            scaler_list.append(scaler)

            # use the trained model to predict the validation lifetime dataset:
            # load the validation lifetime data from the object:
            validationset = self.validationdata
            # extract the validation lifetime data:
            # create a list to select X columns: if the column string contains cm, then identify it as X.
            select_X_list = []
            for string in validationset.columns.tolist():
                if string[0].isdigit():
                    select_X_list.append(string)
            validationlifetime = validationset[select_X_list]
            # take the log10 and make the name shorter
            X = validationlifetime
            X = np.log10(np.array(X.astype(np.float64)))
            # go through the scaler:
            X_scaled = scaler.transform(X)
            # make the prediction:
            y_predict = selected_model.predict(X_scaled)
            y_predictions.append(y_predict)
        self.firststep_prediction = y_predictions

        return model_list, scaler_list, y_predictions


    def dynamic_simulator(self, fixlist = [['Et_eV_1', 'logSn_1', 'logSp_1'], [0.060040545422049535, -13.0, -15.839891398434329]]):
        '''
        This function will simulate a new dataset.

        input: a dictionary of fixed parameters.

        output: a dataframe of lifetime and defect parameters.
        '''
        # inputs:
        SAVEDIR = r"C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Savedir_example" # you can change this to your own path
        FILEPATH = "advanced_example\\data\\sample_original_L.csv"
        TEMPERATURE = [150, 200, 250, 300, 350, 400] # below 400K
        # DOPING = [1e14, 5e14, 1e15, 5e15, 1e16] # *len(TEMPERATURE) # make sure T and doping have same length
        # DOPING = [1e15]
        DOPING = [1e15] *len(TEMPERATURE) # make sure T and doping have same length
        # if using different doping levels: each temperature will match each doping.
        # so each element of temperature will repeat for length of itself times
        # TEMPERATURE
        WAFERTYPE = 'p'
        NAME = 'advanced example - multi_level_L'
        # define the hyper parameters.
        PARAMETERS = {
                        'name':'advanced example - multi_level_L',
                        'save': False,   # True to save a copy of the printed log, the outputed model and data
                        'logML':False,   #   Log the output of the console to a text file
                        'n_defects': self.simulate_size, # Size of simulated defect data set for machine learning
                        'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
                        'non-feature_col':['Mode','Label',"Name","Et_eV_1","Sn_cm2_1","Sp_cm2_1",'k_1','logSn_1','logSp_1','logk_1','bandgap_1',"Et_eV_2","Sn_cm2_2","Sp_cm2_2",'k_2','logSn_2','logSp_2','logk_2','bandgap_2']
                    }
        PARAM = {
                    'type': 'p',                #   Wafer doping type
                    'Et_min_1':0,             #   Minimum defect energy level
                    'Et_max_1':0.55,              #   Maximum defect energy level
                    'Et_min_2':0,             #   Minimum defect energy level
                    'Et_max_2':0.55,              #   Maximum defect energy level
                    'S_min_1_p':1E-17,              #   Minimum capture cross section for hole.
                    'S_min_1_n':1E-17,          #   Minimum capture cross section for electron.
                    'S_max_1_p':1E-13,              #   Maximum capture cross section for hole.
                    'S_max_1_n':1E-13,              # maximum capcture cross section for electron.
                    'S_min_2_p':1E-17,              #   Minimum capture cross section for hole.
                    'S_min_2_n':1E-17,          #   Minimum capture cross section for electron.
                    'S_max_2_p':1E-13,              #   Maximum capture cross section for hole.
                    'S_max_2_n':1E-13,              # maximum capcture cross section for electron.
                    'Nt':1E12,                  #   Defect density
                    'check_auger':True,     #   Check wether to resample if lifetime is auger-limited
                    'noise':'',             #   Enable noiseparam
                    'noiseparam':0,         #   Adds noise proportional to the log of Delta n
                    }

        # update hte PARAM using if statement:
        counter = 0
        for param_name in fixlist[0]:
            # print(counter)
            if param_name == 'Et_eV_1':
                PARAM.update({'Et_min_1':fixlist[1][counter], 'Et_max_1':fixlist[1][counter]})
            elif param_name == 'logSn_1':
                PARAM.update({'S_min_1_n':10**(fixlist[1][counter]), 'S_max_1_n':10**(fixlist[1][counter])})
            elif param_name == 'logSp_1':
                PARAM.update({'S_min_1_p':10**(fixlist[1][counter]), 'S_max_1_p':10**(fixlist[1][counter])})
            elif param_name == 'Et_eV_2':
                PARAM.update({'Et_min_2':fixlist[1][counter], 'Et_max_2':fixlist[1][counter]})
            elif param_name == 'logSn_2':
                PARAM.update({'S_min_2_n':10**(fixlist[1][counter]), 'S_max_2_n':10**(fixlist[1][counter])})
            elif param_name == 'logSp_2':
                PARAM.update({'S_min_2_p':10**(fixlist[1][counter]), 'S_max_2_p':10**(fixlist[1][counter])})
            # update the counter:
            counter = counter + 1

        # define the experiement.
        exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
        # simulate single two level lifetime data.
        db_sah=DPML.generateDB_sah(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM) # one two-level defect data
        # print(db_sah)
        db_sah['Mode']=['Single two-level']*len(db_sah)
        dataDf=db_sah
        dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
        exp.uploadDB(dataDf)
        vocab={
            '0':'Two one-level',
            '1':'Single two-level',
        }
        # export the data.
        # exp.exportDataset()
        # print(dataDf)
        return dataDf


    def t_step_train_predict(self):
        '''
        This function takes in the prediction from the first step, then dynamically generate the new dataset
        to train the new model then predict the second step parameters.
        '''
        print('The dynamic regressor chain is ' + str(self.task))

        # make the first step prediction from the given dataset:
        model_list, scaler_list, y_predictions_1 = self.datatraining(self.first_step_training_path, self.n_repeat, self.task[0])

        # read off the task name from first step:
        tasks1 = self.task[0]

        # extract the validation lifetime data:
        validationset = self.validationdata
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in validationset.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        validationset = validationset[select_X_list]

        # print(np.shape(y_predictions_1))
        # now we got the first step prediction y_predictions_1 with dimension [first step tasks]*[datasize] (3, 8)
        # create emtply list to collect the prediction for each validation point.
        y_predictions_2 = []
        # iterate for each prediction from validation set:
        counter = 0
        for prediction in np.transpose(y_predictions_1):
            # update the counter.
            counter = counter + 1
            # print which validation point we are up to and how many are there in total
            print('the validation data size is ' + str(np.shape(self.validationdata)[0]))
            print('generating data for validation data point ' + str(counter))
            validationpoint = validationset.iloc[counter-1, :]
            # print(prediction.tolist())
            # simulate the new dataset with the fixed first step prediction values.
            fixlist = [tasks1, prediction.tolist()]
            data2 = self.dynamic_simulator(fixlist = fixlist)
            # print(fixlist) this is correct
            # create empty list to collect prediction for each task.'
            y_predictions = []
            # iterate through each second step ML tasks:
            for task2 in self.task[1]:
                print('for task ' + str(task2))
                # train the ML model for this task, define the maching learning object.
                training_step2 = MyMLdata_2level(self.first_step_training_path, 'bandgap1', self.n_repeat)
                # update the step 2 training data as data2
                training_step2.data = data2
                # update the ML task for second step training.
                training_step2.singletask = task2
                # try to make it return the best R2 score model for all trials all models.
                r2_frame, y_prediction_frame, y_test_frame, selected_model, scaler = training_step2.regression_repeat(output_y_pred=True)
                # apply the same model and scaler on the validation set lifetime:

                # take the log10 and make the name shorter
                X = np.array(validationpoint).reshape(1, -1)
                X = np.log10(np.array(X.astype(np.float64)))
                # go through the scaler:
                X_scaled = scaler.transform(X)
                # make the prediction:
                y_predict = selected_model.predict(X_scaled)
                # print(y_predict) # expect one value # checked out.
                y_predictions.append(y_predict)
            # collect the prediction list for each task into the frame.
            y_predictions_2.append(y_predictions)

            # store the prediction into the object.
            # self.y_predictions_1 = y_predictions_1
            # self.y_predictions_2 = y_predictions_2

        return y_predictions_1, y_predictions_2


    def evaluation(self):
        '''
        This function will evaluate prediction from this object with the original value.
        '''
        # get the prediction results from both steps
        y_predictions_1, y_predictions_2 = self.t_step_train_predict()

        # cascade the two predictions together:
        # cascade the name:
        tasks = list(np.concatenate(self.task).flat)
        # concanate the values:
        # print(np.shape(y_predictions_1))
        # y_predictions_2 = np.reshape(np.array(y_predictions_2).flat, np.shape(y_predictions_1))
        # print(np.shape(y_predictions_2))
        # print(np.array(np.transpose(y_predictions_2)).reshape(np.shape(y_predictions_1)))
        # print(np.array(y_predictions_1))
        y_predictions_1 = np.array(y_predictions_1)
        y_predictions_2 = np.reshape(np.array(y_predictions_2).flat, np.shape(y_predictions_1))
        y_predictions = np.concatenate((y_predictions_1, y_predictions_2))
        # print(np.shape(y_predictions))

        # extract the validations y data:
        y_validation = self.validationdata[tasks]

        print(y_predictions)
        print(y_validation)
        print(y_predictions_1)
        print(y_predictions_2)
