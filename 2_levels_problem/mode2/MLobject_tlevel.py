# %%-- To do list:
'''
1. Figure out the columes of the confusion matrix.
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
from sklearn.inspection import permutation_importance
# %%-


class MyMLdata_2level:

    """
    MyMLdata is an object that is a panda dataframe containing the lifetime data.
    """
# %%--- Initialize the object
    def __init__(self, path, task, repeat, load_data_from_path = True):
        """
        1.  Load the data through the inputting path.
        2.  Define the task:  input as a string.
            k: do regression on k (ratio of the capture cross sectional area.)
            Et_eV: do regression on Et
            bandgap: identify if Et is above or below Ei
        3.  define default parameters for machine learinng
        """
        # define the default maching learning setting for both regression and classification.
        # regression_default_param = {
        # 'model_names': ['KNN', 'Ridge Linear Regression', 'Random Forest' , 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'], # a list of name for each model.
        # 'model_lists': [KNeighborsRegressor(), Ridge(), RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1), MLPRegressor(((100, 300, 500, 700, 500, 300, 100)),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive'), GradientBoostingRegressor(verbose=0,loss='ls',max_depth=10), AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='linear'), SVR(kernel='rbf',C=5,verbose=0, gamma="auto")],# a list of model improted from sklearn
        # 'gridsearchlist': [True, True, False, False, False, False, False], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        # 'param_list': [{'n_neighbors':range(1, 30)}, {'alpha': [0.01, 0.1, 1, 10]}, {'n_estimators': [200, 100, 1000, 500, 2000], 'verbose':[0], 'n_jobs':[-1]}, {'hidden_layer_sizes':((100, 300, 300, 100), (100, 300, 500, 300, 100), (200, 600, 600, 200), (200, 600, 900, 600, 200), (100, 300, 500, 700, 500, 300, 100)), 'alpha': [0.001], 'learning_rate':['adaptive']}, {'n_estimators':[200, 100]}, {'n_estimators':[50, 100]}, {'C': [0.1, 1, 10], 'epsilon': [1e-2, 0.1, 1]}]# a list of key parameters correspond to the models in the model_lists if we are going to do grid searching
        # }
        # only neural network.
        # regression_default_param = {
        # 'model_names': ['Neural Network'], # a list of name for each model.
        # 'model_lists': [MLPRegressor(((100, 300, 500, 700, 500, 300, 100)),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        # 'gridsearchlist': [True], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        # 'param_list': [{'hidden_layer_sizes':((100, 300, 300, 100), (100, 300, 500, 300, 100), (200, 600, 600, 200), (200, 600, 900, 600, 200), (100, 300, 500, 700, 500, 300, 100)), 'alpha': [0.001], 'learning_rate':['adaptive']}# a list of key parameters correspond to the models in the model_lists if we are going to do grid searching
        # ]}
        # only the quick ones
        # regression_default_param = {
        # 'model_names': ['KNN', 'Ridge Linear Regression'], # a list of name for each model.
        # 'model_lists': [KNeighborsRegressor(), Ridge()],# a list of model improted from sklearn
        # 'gridsearchlist': [True, True], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        # 'param_list': [{'n_neighbors':range(1, 30)}, {'alpha': [0.01, 0.1, 1, 10]}]
        # }
        # only the decision tree based estimators:
        # regression_default_param = {
        # 'model_names': ['Random Forest', 'Gradient Boosting', 'Ada Boosting'], # a list of name for each model.
        # 'model_lists': [RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1), GradientBoostingRegressor(verbose=0,loss='ls',max_depth=10), AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='linear')],# a list of model improted from sklearn
        # 'gridsearchlist': [False, False, False], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        # 'param_list': [{'n_estimators': [200, 100, 1000, 500, 2000]}, {'n_estimators':[200, 100]}, {'n_estimators':[50, 100]}]# a list of key parameters correspond to the models in the model_lists if we are going to do grid searching
        # }
        # random forest only
        regression_default_param = {
        'model_names': ['Random Forest'], # a list of name for each model.
        'model_lists': [RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)],# a list of model improted from sklearn
        'gridsearchlist': [False], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        'param_list': [{'n_estimators': [200, 100, 1000, 500, 2000]}]# a list of key parameters correspond to the models in the model_lists if we are going to do grid searching
        }
        # random forest and linear regression only:
        # regression_default_param = {
        # 'model_names': ['Random Forest', 'Ridge Linear Regression'], # a list of name for each model.
        # 'model_lists': [RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1), Ridge()],# a list of model improted from sklearn
        # 'gridsearchlist': [False, False], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        # 'param_list': [{'n_estimators': [200, 100, 1000, 500, 2000]}, {'alpha': [0.01, 0.1, 1, 10]}]
        # }
        # # all classification models:
        # classification_default_param = {
        # 'model_names': ['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'], # a list of name for each model.
        # 'model_lists': [KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1), GradientBoostingClassifier(verbose=0,loss='deviance'), AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10), GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        # 'gridsearchlist': [False, False, False, False, False, False, False, False],
        # 'param_list': [{'n_neighbors':range(1, 30)}, {'C': [0.1, 1, 10], 'kernel': ('linear', 'poly', 'rbf')},  {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]},{'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        # }
        # # without SVC:
        # classification_default_param = {
        # 'model_names': ['KNN', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'], # a list of name for each model.
        # 'model_lists': [KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1), GradientBoostingClassifier(verbose=0,loss='deviance'), AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10), GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        # 'gridsearchlist': [False, False, False, False, False, False, False],
        # 'param_list': [{'n_neighbors':range(1, 30)}, {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]},{'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        # }
        # NN compared to Naive bias only only:
        # classification_default_param = {
        # 'model_names': ['Naive Bayes', 'Neural Network'], # a list of name for each model.
        # 'model_lists': [GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        # 'gridsearchlist': [False, False],
        # 'param_list': [{'var_smoothing':[1e-9, 1e-3]}, {'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        # }
        # wihtout CSV, gradient boosting, adabosting
        classification_default_param = {
        'model_names': ['KNN', 'Decision tree', 'Random Forest',  'Naive Bayes', 'Neural Network'], # a list of name for each model.
        'model_lists': [KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1), GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        'gridsearchlist': [False, False, False, False, False, False, False],
        'param_list': [{'n_neighbors':range(1, 30)}, {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        }

        self.path = path
        self.data = pd.read_csv(path)
        self.singletask = task
        self.repetition = repeat
        self.reg_param = regression_default_param
        self.cla_param = classification_default_param
        self.regression_matrix = 'R2'
# %%-


# %%--- Regression machine learning tasks.


    def transparency_calculator(self, datasize):
        '''
        This function will calcualte a suitable data transparency given the datasize for a scatter plot.

        input: datasize: an integer.
        '''
        if datasize>800:
            alpha = 800/datasize*0.5
        else:
            alpha = 0.5
        return alpha


    def regression_repeat(self, plot=False, output_y_pred=False):
        # extract the X and y from previous step: here X is log(lifetime)
        X, y = self.pre_processor()
        # n_repeat is the number of reeptition for this task
        n_repeat = self.repetition
        """
        What it does:
        1. do the train test split for X and y
        2. scale the X features for machine learning
        3. Send the X and y after splitting and scaling into the regression function
        4. Repeat the above steps
        5. Keep track of the r2 score after each training.

        input:
            plot: if True, then predicted vs real will be plotted for each model after each training. if False, it will not plot anything.
            output_y_pred: if True, it will output both the prediction value and the trained model.
        output:
            r2_frame: a dataframe, each row correspond to a trial and each column correspond to a model name.
            Also plot a boxplot of different model's R2 score
            Also plot a real vs prediction for the best trial best model.
            y_prediction_frame: the y prediction.
            y_test_frame: the y test set
            best_model: the model that has the best average score, since we have multiple trials, it will return the first trial model.
            scaler_return: the scaler associated with the first trial for the best model.
        """


        # set up counter to count the number of repetition
        counter = 0
        # create an emptly list to collect the r2 and mean absolute error values for each trials
        r2_frame = []
        meanabs_frame = []
        y_prediction_frame = []
        y_test_frame = []
        trained_model_frame = []
        # iterate for each repeatition:
        while counter < n_repeat:
            # update the counter
            counter = counter + 1
            # pro process the data:
            # if self.singletask != 'Et_minus':
            #     y = y/np.abs(np.max(y))
            # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            # scale the data:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            # when asked to return the best model, we need to return its corresponding scaler as well, since the best model will be the first trial, return the scaler for first trial as well:
            if counter == 1:
                scaler_return = scaler # return the first iteration scaler
            # we must apply the scaling to the test set that we computed for the training set
            X_test_scaled = scaler.transform(X_test)
            # train the different models and collect the r2 score.
            # if output_y_pred == True: # if we plan to collect the y predction
            r2score, mae_list, y_prediction, y_test, trained_model_list = self.regression_training(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train, y_test=y_test, plot=plot, output_y_pred=True)
            r2_frame.append(r2score)
            meanabs_frame.append(mae_list) # the dimension is repeatition * different models.
            y_prediction_frame.append(y_prediction)
            y_test_frame.append(y_test)
            trained_model_frame.append(trained_model_list)
            # else: # when we do not need to collect the y prediction
            # r2score = self.regression_training(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train, y_test=y_test, plot=plot)
            # r2_frame.append(r2score)
            # print the number of iteration finished after finishing each iteration
            print('finish iteration ' + str(counter))

        # now r2_frame is a list of list containing the values for each trial for each model.
        # convert it into dataframe for box plot.
        r2_frame = pd.DataFrame(r2_frame, columns=self.reg_param['model_names'])
        r2_av = np.average(r2_frame, axis=0)
        r2_std = np.std(r2_frame, axis=0)
        labels = []
        for k in range(len(r2_av)):
            labels.append(str(r2_frame.columns[k] +' ('+ str(round(r2_av[k], 3)) + r'$\pm$' + str(round(r2_std[k], 3)) + ')'))
        # box plot the data.
        plt.figure()
        plt.boxplot(r2_frame, vert=False, labels=labels)
        plt.title('$R^2$ scores for ' + str(self.singletask))
        # append the data label for the boxplot
        # for k in range(len(r2_av)):
        #     y = 8.5/(len(r2_av) + 1)*k + 0.5
        #     # x=0.99
        #     plt.text(x=0.98, y=y, s=str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)))
        plt.show()

        # do the boxplot for mean absolute error.
        # convert it into dataframe for box plot.
        meanabs_frame = pd.DataFrame(meanabs_frame, columns=self.reg_param['model_names'])
        meanabs_av = np.average(meanabs_frame, axis=0)
        meanabs_std = np.std(meanabs_frame, axis=0)
        labels = []
        for k in range(len(meanabs_av)):
            if self.singletask[0] == 'E':
                unit = 'eV'
            else:
                unit = '(log of k)'
            labels.append(str(meanabs_frame.columns[k] +' ('+ str(round(meanabs_av[k], 3)) + r'$\pm$' + str(round(meanabs_std[k], 3)) + ')') + unit)
        # box plot the data.
        plt.figure()
        plt.boxplot(meanabs_frame, vert=False, labels=labels)
        plt.title('Mean absolute error scores for ' + str(self.singletask))
        # append the data label for the boxplot
        # for k in range(len(r2_av)):
        #     y = 8.5/(len(r2_av) + 1)*k + 0.5
        #     # x=0.99
        #     plt.text(x=0.98, y=y, s=str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)))
        plt.show()

        # plot real vs predicted for the best trial
        r2_score_k = np.array(r2_frame)
        mae_score_k = np.array(meanabs_frame)
        if self.regression_matrix == 'R2':
            # find the position which has the best R2 score.
            max_position = np.argwhere(r2_score_k == np.max(r2_score_k))
            repeat_num = int(max_position[0][0])
            model_num = int(max_position[0][1])
            print('The best performance is decided by highest R2')
        elif self.regression_matrix == 'Mean Absolute Error':
            # find the position which has the best R2 score.
            max_position = np.argwhere(mae_score_k == np.min(mae_score_k))
            repeat_num = int(max_position[0][0])
            model_num = int(max_position[0][1])
            print('The best performance is decided by lowest Mean Absolute Error')
        # plot the graph for real vs predicted
        plt.figure(facecolor='white')
        # print(np.shape(r2_frame))
        # print(np.shape(y_prediction_frame))
        # print(np.shape(y_test_frame))
        # print(np.shape(y_prediction_frame))
        # calculate the transparency:
        alpha=self.transparency_calculator(len(np.array(y_test_frame)[repeat_num]))
        print('transparency of scattering plot is ' + str(alpha))
        plt.scatter(np.array(y_test_frame)[repeat_num], np.array(y_prediction_frame)[repeat_num, :, model_num], label=('$R^2$' + '=' + str(round(np.max(r2_score_k), 3))) + ('  Mean Absolue error' + '=' + str(round(np.min(mae_score_k), 3))), alpha=alpha)
        plt.xlabel('real value')
        plt.ylabel('predicted value')
        plt.title('real vs predicted at trial ' + str(repeat_num + 1) + ' using method ' + str(self.reg_param['model_names'][model_num]) + ' for task ' + str(self.singletask))
        plt.legend(loc=3, framealpha=0.1)
        plt.savefig(str(self.singletask) + '.png')
        plt.show()

        if output_y_pred == False:
            return r2_frame

        else:
            # the dimension of the model frame is [different models][trials]
            # pick the model that has the highest average score.
            best_model = trained_model_frame[0][model_num]
            return r2_frame, y_prediction_frame, y_test_frame, best_model, scaler_return


    def regression_training(self, X_train_scaled, X_test_scaled, y_train, y_test, plot=False, output_y_pred=False):

        """
        input:
            X_train_scaled, X_test_scaled, y_train, y_test
            plot: a boolean input, if True then it will plot real vs predicted for each model
            output_y_pred: a boolean input, if True then it will output the predicted y

        what it does: use the given data to train different regression algarisms (only once)

        output: a list of R2 scores for each model corresponding to 'KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'
        """
        # print(X_train_scaled)
        # use a for loop to train and evaluate each model: firstly read the setting from the object itself.
        model_names = self.reg_param['model_names']
        model_lists = self.reg_param['model_lists']
        gridsearchlist = self.reg_param['gridsearchlist']
        param_list  = self.reg_param['param_list']

        # prepare an empty list to collect the test y
        y_test_list = []
        # prepare an emtply list to collect the predicted y
        y_pred_list = []
        # prepare an emtply list to collect r2 scores:
        r2_list = []
        # prepare an empty list to collect the mean absolute errors.
        mae_list = []
        # prepare an empty list to collect the trained models.
        model_trained_list = []
        # train everything in a for loop,  for each model.
        for modelindex in range(np.shape(model_names)[0]):
            # read the name, model and parameter from the lists
            name = model_names[modelindex]
            # print(name)
            model = model_lists[modelindex]
            # print(model)
            param = param_list[modelindex]
            # print(param)
            gridsearch = gridsearchlist[modelindex]
            # print('whether use grid search: ' + str(gridsearch))
            if gridsearch==True:
                # define the grid search object
                grid = GridSearchCV(model, param)
                # train the grid search object: if it is neural network, use the scaled y data
                grid.fit(X_train_scaled, y_train)
                # use the trained model to predict the y
                y_pred = grid.predict(X_test_scaled)
                # collect the trained model.
                model_trained_list.append(grid)
            else:
                # just use the original model.
                model.fit(X_train_scaled, y_train)
                # predict with the original model using defalt settings
                y_pred = model.predict(X_test_scaled)
                # collect the trained model.
                model_trained_list.append(model)
            # collect the y values
            y_pred_list.append(y_pred)
            y_test_list.append(y_test)
            # evaluate the model using R2 score:
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            # evaluate the model using mean absolute errors
            mae = mean_absolute_error(y_test, y_pred)
            mae_list.append(mae)
            # print the output
            print('finish training ' + name + ', the ' + 'R2' + ' score is ' + str(r2))
            print('The mean absolute error is ' + str(mae))
            # plot the real vs predicted graph if needed
            if plot==True:
                plt.figure()
                plt.scatter(y_test, y_pred, marker='+')
                plt.xlabel('real value')
                plt.ylabel('predicted')
                plt.title('predicted vs real for ' + name)
                plt.show()

        # output the list of list as a datagrame and name them:
        y_output = pd.DataFrame(np.transpose(np.array(y_pred_list)))
        # put the name on it:
        y_output.columns = model_names
        # output hte prediction only if necessary:
        if output_y_pred == True:
            return r2_list, mae_list, y_output, y_test, model_trained_list
        # this function will return all 2r scores and mean absolute errors
        else:
            return r2_list, mae_list

# %%-


# %%--- Classification machine learning tasks.
    def classification_training(self, X_train_scaled, X_test_scaled, y_train, y_test, display_confusion_matrix=False, return_model=False):
        """
        This function is only capable for binary classification yet.
        input:
            X_train_scaled, X_test_scaled, y_train, y_test
            display_confusion_matrix: a boolean input if True then each time finish trianing a model will display its corresponding confusion matrix.

        output: a list of accuracy for each model corresponding to 'KNN', 'SVC', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Naive Bayes', 'Neural Network'
        """
        # use a for loop to train and evaluate each model:
        model_names = self.cla_param['model_names']
        model_lists = self.cla_param['model_lists']
        gridsearchlist = self.cla_param['gridsearchlist']
        param_list  = self.cla_param['param_list']
        # prepare an emtply list to collect f1 scores:
        f1_list = []
        # prepare list to collect y test
        y_test_list = []
        # prepare an emtply list to collect the predicted y
        y_pred_list = []
        # train everything in a for loop
        # try to find the best model
        bestmodel = model_lists[0]
        for modelindex in range(np.shape(model_names)[0]):
            # read the name, model and parameter from the lists
            name = model_names[modelindex]
            # print(name)
            model = model_lists[modelindex]
            # print(model)
            param = param_list[modelindex]
            # print(param)
            gridsearch = gridsearchlist[modelindex]
            # print('whether use grid search: ' + str(gridsearch))
            if gridsearch==True:
                # define the grid search object
                model = GridSearchCV(model, param)
                # train the grid search object: if it is neural network, use the scaled y data
                y_train = np.array(y_train)
                model.fit(X_train_scaled, y_train)
                # use the trained model to predict the y
                y_pred = grid.predict(X_test_scaled)
            else:
                # just use the original model.
                model.fit(X_train_scaled, y_train)
                # predict with the original model using defalt settings
                y_pred = model.predict(X_test_scaled)

            # evaluate the model using micro f1 score (accuracy):
            f1 = f1_score(y_test, y_pred, average='macro')
            f1_list.append(f1)
            y_pred_list.append(y_pred)
            y_test_list.append(y_test)
            # print the output
            print('finish training ' + name + ', the accuracy is ' + str(f1))
            # display the confusion matrix
            if display_confusion_matrix==True:
                print(confusion_matrix(y_test, y_pred, normalize='all'))
            # update the best model.
            if f1 == np.max(f1_list):
                bestmodel = model

        if return_model == True:
            # return the model of the best performance
            return bestmodel, f1_list, y_pred_list, y_test_list
        else:
            return f1_list, y_pred_list, y_test_list


    def classification_repeat(self, display_confusion_matrix=False, return_model=False):
        """
        input:
            df: the dataframe to work on
            n_repeat: the number of repetition needed
        output:
            f1_frame: a dataframe of f1 score.
            Also plotted a boxplot of f1 score for each model.
        """
        # use the pre processor to get X and y
        X, y = self.pre_processor()
        # n_repeat is the number of repetition for this task.
        n_repeat = self.repetition
        # set up counter to count the number of repetition
        counter = 0
        # create an emptly list to collect the f1 and mean absolute error values for each trials
        f1_frame = []
        y_prediction_frame = []
        y_test_frame = []
        while counter < n_repeat:
            # update the counter
            counter = counter + 1
            # pro process the data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            # scale the data:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            # we must apply the scaling to the test set that we computed for the training set
            X_test_scaled = scaler.transform(X_test)
            bestmodel, f1_score, y_pred, y_test= self.classification_training(X_train_scaled, X_test_scaled, y_train, y_test, return_model=True, display_confusion_matrix=display_confusion_matrix)
            f1_frame.append(f1_score)
            y_prediction_frame.append(y_pred)
            y_test_frame.append(y_test)
            # print the number of iteration finished after finishing each iteration
            print('finish iteration ' + str(counter))

        # now f1_frame is a list of list containing the values for each trial for each model.
        # convert it into dataframe for box plot.
        f1_frame = pd.DataFrame(f1_frame, columns=self.cla_param['model_names'])
        f1_av = np.average(f1_frame, axis=0)
        f1_std = np.std(f1_frame, axis=0)
        labels = []
        for k in range(len(f1_av)):
            labels.append(str(f1_frame.columns[k] +' ('+ str(round(f1_av[k], 3)) + r'$\pm$' + str(round(f1_std[k], 3)) + ')'))
        # box plot the data.
        plt.figure(facecolor='white')
        plt.boxplot(f1_frame, vert=False, labels=labels)
        plt.title('$F_1$' + 'score for classification ' + str(self.singletask))
        plt.savefig(str(self.singletask) + '.png')
        plt.show()

        # print hte confusion matrix for the best trial.
        f1_score = np.array(f1_frame)
        max_position = np.argwhere(f1_score == np.max(f1_score))
        repeat_num = int(max_position[0][0])
        model_num = int(max_position[0][1])
        # display the confusion matrix.
        print('The best accuracy is ' + str(round(np.max(f1_score), 3)))
        # print(y_test_frame)
        # print(y_prediction_frame)
        print(confusion_matrix(np.array(y_test_frame)[repeat_num, model_num, :], np.array(y_prediction_frame)[repeat_num,  model_num, :], normalize='all'))

        if return_model == True:
            return f1_frame, y_prediction_frame, y_test_frame, bestmodel, scaler
        else:
            return f1_frame, y_prediction_frame, y_test_frame
# %%-


# %%--- The functions to perform machine learning tasks (the function you will end up calling in another file)
    def perform_singletask_ML(self, plot_graphs=False):
        """
        This is the overall function to perform machine learning for a single task using the other functions

        Input: plot_graphs a boolean input, if true then the function will plot more detail graph after each training.
        Output: three list containing the score for overall, plus and minus.

        What it does:
        1. identify what job is it doing.
        2. perform the maching learning process.
        3. print the evaluation
        """

        if self.singletask == 'logk_1':
            # if the task is to do regression using k
            # apply the regression repeat function for k
            r2_score_k, y_pred_k, y_test_k = self.regression_repeat(plot=plot_graphs, output_y_pred=True)
            # find the position which has the best R2 score.
            r2_score_k_output = r2_score_k
            r2_score_k = np.array(r2_score_k)
            max_position = np.argwhere(r2_score_k == np.max(r2_score_k))
            repeat_num = int(max_position[0][0])
            model_num = int(max_position[0][1])
            # plot the graph for real vs predicted
            plt.figure()
            plt.scatter(np.array(y_test_k)[repeat_num, :], np.array(y_pred_k)[repeat_num, :, model_num], label=('$R^2$' + '=' + str(round(np.max(r2_score_k), 3))))
            plt.xlabel('real k')
            plt.ylabel('predicted k')
            plt.title('real vs predicted at trial ' + str(repeat_num + 1) + ' using method ' + str(self.reg_param['model_names'][model_num]))
            plt.legend()
            plt.show()

            return r2_score_k_output



        elif self.task == 'Et_eV':
            # if the task is to do regression for Et
            # apply the regression for Et plus and Et minus separately
            self.singletask = 'Et_plus'
            # collect the best Et_plus y prediction.
            r2_scores_plus, y_pred_plus, y_test_plus = self.regression_repeat(plot=plot_graphs, output_y_pred=True)
            print('finish training upper bandgap')
            # do the same for Et_minus
            self.singletask = 'Et_minus'
            # collect the best Et_plus y prediction.
            r2_scores_minus, y_pred_minus, y_test_minus = self.regression_repeat(plot=plot_graphs, output_y_pred=True)
            print('finish training lower bandgap')
            # then plot the prediction on both positive and negative side of Et on the same plot.
            # put the upper and lower band data together
            y_pred_together = np.concatenate((y_pred_plus, y_pred_minus), axis=1)
            y_test_together = np.concatenate((y_test_plus, y_test_minus), axis=1)
            # since y_pred_together include an extra dimension for each model, we tile y test to make it has the same shape
            # prepare a list to collect the r2 r2_scores.
            r2_Et = []
            # use a for loop to loopt at each repeatition plot the real vs predicted
            for n in range(np.shape(y_test_together)[0]):
                # prepare a list to collect r2 scores for each model
                r2_model = []
                # loop around each machine learning method as well
                for k in range(np.shape(y_pred_together)[-1]):
                    # calculate the r2 score for each trials
                    r2score = r2_score(y_test_together[n, :], y_pred_together[n, :, k])
                    r2_model.append(r2score)
                r2_Et.append(r2_model)
            # now we have a list of list.
            # convert the list of list into a dataframe.
            r2_Et = pd.DataFrame(r2_Et, columns=self.reg_param['model_names'])
            r2_av = np.average(r2_Et, axis=0)
            r2_std = np.std(r2_Et, axis=0)
            labels = []
            for k in range(len(r2_av)):
                labels.append(str(r2_Et.columns[k] +' ('+ str(round(r2_av[k], 3)) + r'$\pm$' + str(round(r2_std[k], 3)) + ')'))
            # plot the r2 scores as a boxplot
            plt.figure()
            plt.boxplot(r2_Et, vert=False, labels=labels)
            plt.title('$R^2$ scores for $E_t$ regression')
            plt.show()

            # if plot_graphs:
            # find which one has the largest r2 and plot the real vs predicted for the best prediction.
            r2_Et_output = r2_Et
            r2_Et = np.array(r2_Et)
            max_position = np.argwhere(r2_Et == np.max(r2_Et))
            repeat_num = max_position[0][0]
            model_num = max_position[0][1]
            # plot the graph for real vs predicted
            plt.figure()
            plt.scatter(y_test_together[repeat_num, :], y_pred_together[repeat_num, :, model_num], label=('$R^2$' + '=' + str(round(np.max(r2_Et), 3))))
            plt.xlabel('real ' + '$E_t$' + ' (eV)')
            plt.ylabel('predicted' + '$E_t$' + ' (eV)')
            plt.title('real vs predicted at trial ' + str(repeat_num + 1) + ' using method ' + str(self.reg_param['model_names'][model_num]))
            plt.legend()
            plt.show()

            return r2_Et_output, r2_scores_plus, r2_scores_minus

        elif self.task == 'bandgap':
            self.singletask = self.task
            f1_score, y_pred_bg, y_test_bg = self.classification_repeat(display_confusion_matrix=plot_graphs, output_y_pred=True)
            # fine the best f1 score position.
            f1_output = f1_score
            f1_score = np.array(f1_score)
            max_position = np.argwhere(f1_score == np.max(f1_score))
            repeat_num = int(max_position[0][0])
            model_num = int(max_position[0][1])
            # display the confusion matrix.
            print('The best accuracy is ' + str(round(np.max(f1_score), 3)))
            print(confusion_matrix(np.array(y_test_bg)[repeat_num, model_num, :], np.array(y_pred_bg)[repeat_num,  model_num, :], normalize='all'))

            return f1_output


    def perform_alltasks_ML(self, plot_graphs=False, playmusic=False):
        """
        This is the function to perform all tasks together using ML

        Input: plot_graphs a boolean input, if true then the function will plot more detail graph after each training.

        What it does: perform single task ML for each task
        """
        score_list = []
        for tasks in ['k', 'Et_eV', 'bandgap']:
            print('performing prediction for ' + tasks)
            self.task = tasks
            score = self.perform_singletask_ML(plot_graphs=plot_graphs)
            score_list.append(score)
            print('finish predicting ' + tasks)

        # play a nice reminder music after finishing
        if playmusic == True:
            playsound('spongbob.mp3')

        return score_list
# %%-


# %%--- The functions for data visualization
    def mypairplot(self, plot_col):
        """
        This file will plot the pairplot for the chosen data given by plot_col.

        input: plot_col is a list of text containing the column names that we wonna plot.
        """
        # load the data from object.
        df = pd.DataFrame(self.data)
        # extract the ones we wonna plot.
        dfplot = df[plot_col]
        # do the pairplot:
        figure = sn.pairplot(dfplot)


    def mypairplotX(self, plot_col):
        '''
        This file will plot the pariplot for the chosen data frame and the label will be in mathematical text
        note: this function only works for plotting X data against another X data.

        input: plot_col, a list of text containing the colume names that we wonna plot.

        '''
        # load the data from the object
        df = pd.DataFrame(self.data)
        # print(np.shape(df))
        # print(df)
        dfplot = df[plot_col]
        # to convert the numbers into scientific notation to be displayed during plotting, prepare an emptly list to collect the text after conversion
        plot_sci_col = []
        for text in plot_col[:-1]:
            # we need to write the numbers in scientific notation
            # first we split hte text with _
            temp, doping_density, excess_carrier_density = text.split('_')
            # for the converting ones, get rid of the unit
            doping_density = doping_density.split('c')[0]
            excess_carrier_density = excess_carrier_density.split('c')[0]
            # convert the doping and excess carrier density into scientific notation
            doping_sce = "{0:.2E}".format(Decimal(doping_density))
            excess_sce = "{0:.2E}".format(Decimal(doping_density))
            # now put all the text back together and put into the new text list.
            plot_sci_col.append(str(temp) + str(doping_sce) + '$cm^{-3}$' + str(excess_sce) +'$cm^{-3}$')
        # now we got a list of name to display columns for X, add the Et
        plot_sci_col.append('Et_eV')
        dfplotT.columns = plot_sci_col
        sn.set(font_scale=0.8)
        figure = sn.pairplot(dfplotT)


    def C_visiaulization_singeT_singledope(self):
        """
        The aim of this function is to plot the histogram of C1n, C1d on the same plot with x axis logscale

        This function only works if the data only have one temperature and one doping
        """
        # read off the C values using prewritten functions.
        C1n_list, C1d_list, C2n_list, C2d_list = self.C1_C2_C3_C4_calculator(return_C=True)
        # print(np.shape(C3_list)) # expect (8000,)
        # read off the temperautre, doping and excess carrier concentration.
        variable_type, temp_list, doping_level, excess_dn = self.reader_heading()
        for column_index in range(len(list(self.data.columns))):
            # if that column is a variable instead of y
            if variable_type[column_index] == 'X':
                # convert the readed text into numbers for dn and doping:
                doping_level[column_index] = float(doping_level[column_index].split('c')[0])
                excess_dn[column_index] = float(excess_dn[column_index].split('c')[0])
                T = float(temp_list[column_index].split('K')[0])

                doping = doping_level[column_index]
                dn = excess_dn[column_index]
                break

        # start plotting the histogram for Cn
        bins=1000
        plt.figure()
        plt.hist(np.log10(np.array(C1n_list)/np.array(C2n_list)), bins=bins, label='C1d')
        # plt.xscale('log')
        plt.title('Distribution of ratio for $C_{1n}/C_{2n}$')
        plt.xlabel('log10 of the ratio')
        plt.show()

        # start plotting the histogram for Cn
        bins=1000
        plt.figure()
        plt.hist(np.log10(np.array(C1d_list)/np.array(C2d_list)), bins=bins, label='C1d')
        # plt.xscale('log')
        plt.title('Distribution of ratio for $C_{1d}/C_{2d}$')
        plt.xlabel('log10 of the ratio')
        plt.show()

        # # boxplot.
        # # make C into a frame
        # Cframe = pd.DataFrame(np.transpose(np.array([C1n_list, C2n_list])), columns=['$C_1n$', '$C_2n$'])
        # # print(np.shape(Cframe)) # expect (8000,2)
        # plt.figure()
        # plt.boxplot(Cframe, labels=['$C_{1n}$', '$C_{2n}$'])
        # plt.yscale('log')
        # plt.title('comparison between the $C_{n}$ on the numerator')
        # plt.show()
        #
        # # # plot this boxplot of hte ratio for Cn
        # bins=1000
        # ratio_array = np.array(C1n_list)/np.array(C2n_list)
        # plt.figure()
        # plt.boxplot(ratio_array, labels=['${C_{1n}}/{C_{2n}}$'])
        # plt.yscale('log')
        # plt.title('Distribution of ratio of $C_{n}$')
        # plt.show()
        #
        # # box plot for Cd
        # Cframe = pd.DataFrame(np.transpose(np.array([C1d_list, C2d_list])), columns=['$C_1d$', '$C_2d$'])
        # # print(np.shape(Cframe)) # expect (8000,2)
        # plt.figure()
        # plt.boxplot(Cframe, labels=['$C_{1d}$', '$C_{2d}$'])
        # plt.yscale('log')
        # plt.title('comparison between the $C_{d}$ on the numerator')
        # plt.show()
        #
        # # # plot this box plot of the ratio for Cd
        # bins=1000
        # ratio_array = np.array(C1d_list)/np.array(C2d_list)
        # plt.figure()
        # plt.boxplot(ratio_array, labels=['${C_{1d}}/{C_{2d}}$'])
        # plt.yscale('log')
        # plt.title('Distribution of ratio of $C_{d}$')
        # plt.show()


    def C_visiaulization(self, variable='C1d/C2d', task_name='histogram of all lifetime', T=300, doping=1e15):
        """
        This function works in general: OK to vary T and doping now.

        input:
        taskname: a string input that defines the task for visalization.
        variable: a string input that defines the variable that we are taking.
        T: if task is to investigate the certain T, then it define the temperaure to look at, unit is in K
        doping: if task is to investigate the certain doping, then it define the doping level to look at, unit is in cm-3
        """
        self.data = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\lifetimedata.csv')
        # calcualte the parameter based on hte input:
        if variable == 'C1d/C2d':
            # instead of calculating C, read it off from temperatry file directory, ignore the first volumn becase it is just a title volume
            C1d_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C1ddata09_29_36.csv').iloc[:,1:]
            C2d_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C2ddata09_29_36.csv').iloc[:,1:]
            # the first 3 rows are headings, we should not divide them
            heading = C2d_frame.iloc[0:3, :]
            # print(heading) # expect first row doping, second row T, third row dn # checked out.
            # divide the C values
            Cset = np.array(C1d_frame)/np.array(C2d_frame) # in this step we divide the heading values as well, so later we need to replace the first 3 rows with the original headings.
            # replace the first 3 rows with headings
            Cset[0:3, :] = heading
            # sanity check:
            # print(np.shape(Cset)) # expect (8003,3600)
            # print(pd.DataFrame(Cset).head()) # expect 3 rows of heading and 2 rows of ratio value, the ratio value is expected to be large. # check out.
        elif variable == 'C1n/C2n':
            C1n_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C1ndata09_29_36.csv').iloc[:,1:]
            C2n_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C2ndata09_29_36.csv').iloc[:,1:]
            # sanity check:
            # print(str(variable))
            heading = C1n_frame.iloc[0:3, :]
            Cset = np.array(C1n_frame)/np.array(C2n_frame)
            Cset[0:3, :] = heading
        # also write down the variable case when we only want to check C
        elif variable == 'C1n':
            C1n_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C1ndata09_29_36.csv').iloc[:,1:]
            Cset = np.array(C1n_frame)
        elif variable == 'C1d':
            Cset = np.array(pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C1ddata09_29_36.csv').iloc[:,1:])
        elif variable == 'C2n':
            Cset = np.array(pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C2ndata09_29_36.csv').iloc[:,1:])
        elif variable == 'C2d':
            Cset = np.array(pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C2ddata09_29_36.csv').iloc[:,1:])


        # plot the histogram of all lifetime curve.
        if task_name == 'histogram of all lifetime':
            T = 'different T'
            doping = 'different doping'
            self.histogram_2D(Cset, variable, T, doping)

        # plot the histogram of a spesific temperature:
        elif task_name == 'histogram at T':
            # the temperaure to be looked up should be predefined
            # select the Cset where the T row is equalt to the given T, where T is the secone row.

            # sanity check:
            # if you set the T be 150K, then you expect the boolean condition to be 1 for first 50(dn)*6(doping)=300 numbers
            # T = 150

            boolean_condition = (Cset[1, :] == T)

            # print(boolean_condition[0:10]) # expect to be 1
            # print(boolean_condition[296: 305]) # expect half 1 half 0
            # print(boolean_condition[-10:]) # expect all 0
            # print(np.sum(boolean_condition)) # expect to be 300
            # checked out.

            Cset = Cset[:,boolean_condition]
            # sanity check:
            # print(Cset[1, :]) # expect all 300K
            # print(np.mean(Cset[1, :])) # expect to be the same as the setting value.
            # print(np.max(Cset[1, :])) # expect to be the same as the setting value.

            # plot its histogram.
            doping = 'different doping'
            self.histogram_2D(Cset, variable, T, doping)

        # plot the histogram of a spesific doping:
        elif task_name == 'histogram at doping':
            # print(Cset[0, :]) # expect doping levels. # checked out.
            boolean_condition = (Cset[0, :] == doping)
            # sanity check:
            # print(boolean_condition) # expect 6*50 ones # checkd out.
            # print(np.sum(boolean_condition)) # expect 300. # checkd out.
            # # expect first 50 rows to be 1 and 300-349 to be 1 as well.
            # print(np.min(boolean_condition[300:350]))  # expect 1
            # print(np.min(boolean_condition[0:50])) # expect 1
            # Cset = Cset[:,boolean_condition]
            # print(Cset[1, :]) # expect all 300K
            # plot its histogram.
            T = 'different T'
            doping = "{:.2e}".format(doping)
            self.histogram_2D(Cset, variable, T, doping)


        elif task_name == 'plot with T':
            # the task is to plot the scatter for different parameters
            # plot the medium under different parameters
            # we want the x axis to be T:
            # for each T we have a whole colume:
            lifetimedata = Cset[3:, :] # the lifetime data start from the 4th row.
            # print(np.shape(lifetimedata))
            # Sanity check:
            # print(pd.DataFrame(lifetimedata).head()) # expect to be lifetime related # checked out.

            # construct a list of T that is available in the dataset:
            T_row = Cset[1, :] # the second row is for Temperature.
            # make it unique:
            T_list = np.unique(T_row)
            # sanity check:
            # print(T_list) # expect: [150, ...400] with a step of 50, length of the list is 6 # checked out.
            medium_list = []
            mean_list = []
            std_list = []
            for T in T_list:
                # find out the lifetime data corresponding to this temperature:
                boolean_condition = (T_row == T)
                # sanity check:
                # print(sum(boolean_condition)) # expect to be 50*6
                # print(boolean_condition[0:10]) # expect to be 1
                # print(boolean_condition[296: 304]) # expect half 1 half 0
                # print(boolean_condition[-10:]) # expect all 0
                # break
                # checked out
                # print(np.shape(lifetimedata))
                lifetimedata_T = lifetimedata[:,boolean_condition]
                # calcualte the medium, mean, and std.
                medium_list.append(np.median(lifetimedata_T))
                mean_list.append(np.mean(lifetimedata_T))
                std_list.append(np.std(lifetimedata_T))

            # plot them on the same graph:
            plt.figure()
            plt.plot(T_list, np.log10(mean_list))
            plt.title('mean value vs T')
            plt.xlabel('Temperature (K)')
            plt.ylabel('log of mean of ' + str(variable))
            plt.show()

            plt.figure()
            plt.plot(T_list, np.log10(std_list))
            plt.title('std vs T')
            plt.xlabel('Temperature (K)')
            plt.ylabel('log of std of ' + str(variable))
            plt.show()

            plt.figure()
            plt.plot(T_list, np.log10(medium_list))
            plt.title('median vs T')
            plt.xlabel('Temperature (K)')
            plt.ylabel('log of median of ' + str(variable))
            plt.show()


        elif task_name == 'plot with doping':
            # the task is to plot the scatter for different parameters
            # plot the medium under different parameters
            # we want the x axis to be T:
            # for each T we have a whole colume:
            lifetimedata = Cset[3:, :] # the lifetime data start from the 4th row.
            # print(np.shape(lifetimedata))
            # Sanity check:
            # print(pd.DataFrame(lifetimedata).head()) # expect to be lifetime related # checked out.

            # construct a list of doping that is available in the dataset:
            doping_row = Cset[0, :] # the first row is for Temperature.
            # make it unique:
            doping_list = np.unique(doping_row)
            # sanity check:
            # print(doping_list) # expect: a list of doping values length of the list is 6 # checked out.
            medium_list = []
            mean_list = []
            std_list = []
            for doping in doping_list:
                # find out the lifetime data corresponding to this temperature:
                boolean_condition = (doping_row == doping)
                # sanity check:
                # print(sum(boolean_condition)) # expect to be 50*6
                # print(boolean_condition[0:10]) # expect to be 1
                # print(boolean_condition[296: 304]) # expect half 1 half 0
                # print(boolean_condition[-10:]) # expect all 0
                # break
                # checked out
                # print(np.shape(lifetimedata))
                lifetimedata_doping = lifetimedata[:,boolean_condition]
                # calcualte the medium, mean, and std.
                medium_list.append(np.median(lifetimedata_doping))
                mean_list.append(np.mean(lifetimedata_doping))
                std_list.append(np.std(lifetimedata_doping))

            # plot them on the same graph:
            plt.figure()
            plt.plot(np.log10(doping_list), np.log10(mean_list))
            plt.title('mean value vs doping')
            plt.xlabel('log of doping value cm-3')
            plt.ylabel('log of mean of ' + str(variable))
            plt.show()

            plt.figure()
            plt.plot(np.log10(doping_list), np.log10(std_list))
            plt.title('std vs doping')
            plt.xlabel('log of doping value cm-3')
            plt.ylabel('log of std of ' + str(variable))
            plt.show()

            plt.figure()
            plt.plot(np.log10(doping_list), np.log10(medium_list))
            plt.title('median vs doping')
            plt.xlabel('log of doping value cm-3')
            plt.ylabel('log of median of ' + str(variable))
            plt.show()


        elif task_name == 'plot with dn':

            lifetimedata = Cset[3:, :] # the lifetime data start from the 4th row.

            # construct a list of doping that is available in the dataset:
            dn_row = Cset[2, :] # the 3rd row is for dn
            # make it unique:
            dn_list = np.unique(dn_row)
            # sanity check:
            # print(dn_list) # expect: a list of dn values length of the list is 50 # checked out.
            medium_list = []
            mean_list = []
            std_list = []
            for dn in dn_list:
                # find out the lifetime data corresponding to this temperature:
                boolean_condition = (dn_row == dn)
                # sanity check:
                # print(sum(boolean_condition)) # expect to be 50*6
                # print(boolean_condition[0:10]) # expect to be 1
                # print(boolean_condition[296: 304]) # expect half 1 half 0
                # print(boolean_condition[-10:]) # expect all 0
                # break
                # checked out
                # print(np.shape(lifetimedata))
                lifetimedata_dn = lifetimedata[:,boolean_condition]
                # calcualte the medium, mean, and std.
                medium_list.append(np.median(lifetimedata_dn))
                mean_list.append(np.mean(lifetimedata_dn))
                std_list.append(np.std(lifetimedata_dn))

            # plot them on different graphs
            plt.figure()
            plt.plot(np.log10(dn_list), np.log10(mean_list))
            plt.title('mean value vs dn')
            plt.xlabel('log of dn value cm-3')
            plt.ylabel('log of mean of ' + str(variable))
            plt.show()

            plt.figure()
            plt.plot(np.log10(dn_list), np.log10(std_list))
            plt.title('std vs dn')
            plt.xlabel('log of dn value cm-3')
            plt.ylabel('log of std of ' + str(variable))
            plt.show()

            plt.figure()
            plt.plot(np.log10(dn_list), np.log10(medium_list))
            plt.title('median vs dn')
            plt.xlabel('log of dn value cm-3')
            plt.ylabel('log of median of ' + str(variable))
            plt.show()

        elif task_name == 'plot with Et1-Et2':

            # read off the Et values from original data:
            Et1 = self.data['Et_eV_1']
            # sanity check:
            # print(Et1[0]) # expect 0.140651095 # checked out.
            Et2 = self.data['Et_eV_2']
            # subtraction:
            Et_diff = np.array(Et1)-np.array(Et2)
            # sanity check:
            # print(np.shape(E_diff)) # expect 8000*1
            # print(E_diff[0]) # expect 0.140651095-0.0111481=0.12955
            # checked out.

            # each row has a unique Et, so take mean, medium and std for each row of lifetime data:
            lifetimedata = Cset[3:, :]
            medium_list = np.median(lifetimedata, axis=1)
            # print(np.shape(medium_list)) # expect 8000*1 # checked out
            mean_list = np.mean(lifetimedata, axis=1)
            std_list = np.std(lifetimedata, axis=1)

            # denoise the data by taking the average:
            Et_diff = self.data_averager(Et_diff)
            medium_list = self.data_averager(medium_list)
            mean_list = self.data_averager(mean_list)
            std_list = self.data_averager(std_list)

            # plot them on different graphs
            plt.figure()
            plt.scatter(Et_diff, np.log10(mean_list), marker='.')
            plt.title('mean value vs Et1-Et2')
            plt.xlabel('Et1-Et2')
            plt.ylabel('log of mean of ' + str(variable))
            plt.show()

            plt.figure()
            plt.scatter(Et_diff, np.log10(std_list), marker='.')
            plt.title('std vs Et1-Et2')
            plt.xlabel('Et1-Et2')
            plt.ylabel('log of std of ' + str(variable))
            plt.show()

            plt.figure()
            plt.scatter(Et_diff, np.log10(medium_list), marker='.')
            plt.title('median vs Et1-Et2')
            plt.xlabel('Et1-Et2')
            plt.ylabel('log of median of ' + str(variable))
            plt.show()

        elif task_name == 'C histogram compare':
            # the purpose is to plot the histogram of C1n and C2n on the same graphs
            # then plot C1d and C2d on the same histogram.

            # extract the parameters:
            C1d_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C1ddata09_29_36.csv').iloc[:,1:]
            C2d_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C2ddata09_29_36.csv').iloc[:,1:]
            C1n_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C1ndata09_29_36.csv').iloc[:,1:]
            C2n_frame = pd.read_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\set11\theCresults\visialization\varyT_varydoping_p_8000\C2ndata09_29_36.csv').iloc[:,1:]
            # get rid of the headings
            C1d = np.array(C1d_frame)[3:, :].flatten()
            C2d = np.array(C2d_frame)[3:, :].flatten()
            C1n = np.array(C1n_frame)[3:, :].flatten()
            C2n = np.array(C2n_frame)[3:, :].flatten()

            # plot the histogram for Cd:
            bins = 10000
            plt.figure()
            plt.hist(np.log10(C1d), bins=bins, label='$C_{1d}$')
            plt.hist(np.log10(C2d), bins=bins, label='$C_{2d}$')
            plt.legend()
            plt.title('Distribution of Cd')
            plt.xlabel('log10 of Cd')
            plt.show()

            # plot the histogram for Cn:
            bins = 10000
            plt.figure()
            plt.hist(np.log10(C1n), bins=bins, label='$C_{1n}$')
            plt.hist(np.log10(C2n), bins=bins, label='$C_{2n}$')
            plt.legend()
            plt.title('Distribution of Cn')
            plt.xlabel('log10 of Cn')
            plt.show()


    def histogram_2D(self, Cset, variable, T, doping):
        """
        This function firstly flatten the 2D data into 1D then plot its histogram.

        input: dataset: should be array like
        """
        # restructure the data back to 1D
        Cset = Cset.flatten()

        # sanity check:
        # print(np.shape(Cset)) # expect 3600*8003

        # plot a histogram: x axis is log10 of the lifetime.
        bins=10000
        plt.figure()
        plt.hist(np.log10(Cset), bins=bins)
        # plt.xscale('log')
        plt.title('Distribution of ratio for ' + str(variable) + ' ' +str(T) + ' ' +str(doping))
        plt.xlabel('log10 of the variable')
        plt.show()


    def data_averager(self, data, length = 1):
        """
        This function will take the average for every length data point and output an array.

        input:
        data: an 1-D array like function.
        length: an integer that you want to take average for every length, it must be able to divide data size.

        output: an arry that is taken average.
        """

        # reshape the data into the array:
        # print(np.size(data))
        data = np.reshape(data, (length,int(np.size(data)/length)))

        # take the average for each row.
        data = np.average(data, axis=0)

        # take the medium for each row.
        # data = np.min(data, axis=0)

        return data


    def feature_importance_visualisation(self, parameter):
        # the plan is to visualize the feature importance of each feature to the given parameter.
        # load the original dataset: don't delete the other parameters (there will be dataledage in the prediction but it's OK)
        fulldata = self.data

        # make the prediction using random forest model.

        # extract the lifetime columns and take log base 10:
        select_X_list = []
        for string in fulldata.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = fulldata[select_X_list] # take the lifetime as X, delete any column that does not start with a number.
        # print(X.where(X==0))
        X = np.log10(X) # take the log of lifetime data.

        # extract the other parameters that are not the parameters:
        known_param = []
        for defect_param in ['Et_eV_2', 'logSn_2', 'logSp_2', 'Et_eV_1', 'logSn_1', 'logSp_1']:
        # if it is not the known parameter.
            if defect_param != parameter:
                # add it into hte X as well.
                X[defect_param] = fulldata[defect_param]
                known_param.append(defect_param)

        print(X.columns.tolist())

        # define the y to be the input parmaeter:
        y = fulldata[parameter]

        # define the scaler:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # scale the data:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        # train the random forest model.
        model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        # output the importance.
        importances = model.feature_importances_

        # still make hte prediction and plot the real vs predicted: we do expect this prediction to be the higher boundary for chain regressoe behaviour.
        y_pred = model.predict(X_test_scaled)
        # compute the evaluation matrixes.
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # plot the real vs predicted.
        plt.figure(facecolor='white')
        # calculate the transparency:
        alpha=self.transparency_calculator(len(y_test))
        print('transparency of scattering plot is ' + str(alpha))
        plt.scatter(y_test, y_pred, label=('$R^2$' + '=' + str(round(r2, 3))) + ('  Mean Absolue error' + '=' + str(round(mae, 3))), alpha=alpha)
        plt.xlabel('real value')
        plt.ylabel('predicted value')
        plt.title('real vs predicted ' + ' using method random forest known all other features' + ' for task ' + str(parameter))
        plt.legend(loc=3, framealpha=0.1)
        # plt.savefig(str(self.singletask) + '.png')
        plt.show()

        # visualize the feature importance:

        # the importance of the lifetime data:
        # extract the lifetime data importances.
        lifetime_importance = importances[:-5]
        # plot the importances of lifetime data:
        plt.figure()
        plt.plot(lifetime_importance)
        plt.show()

        # lets plot the dn vs importances.
        

        # the importance of the other defect parameters:
        # defect_param_importance = importance[-5:]
        # plt.figure()
        #

# %%-


# %%--- The per-processors before machine learning tasks
    def reader_heading(self):
        """
        This function takes the loaded dataframe then return four lists:
        variable_type: a list containing string X or y, if it is X that means this column is lifetime data, if it is y this column is the target values
        temp_list: a list of temperature (string, with units) for each column that labeled X
        dopiong_level: a list of doping levels (string, with units) for each column that labeled X
        excess_dn: a list of excess carrier concentration (string, with units) for each column that labeled X.
        """
        # extract the heading.
        headings = list(self.data.columns)
        # extract the information from headings
        # prepare the empty list to collect temperatures, doping levels and excess carrier concentration
        temp_list = []
        doping_level = []
        excess_dn = []
        variable_type = [] # it will be a list of X and y, if it is X means it is a variable if it is y it means that is the target value we want to predict
        for string in headings:
            # if the first element of the string is not a number, then we know it is a part of y rather than lifetime data:
            if string[0].isdigit() == False:
                variable_type.append('y')
                temp_list.append('Nan')
                doping_level.append('Nan')
                excess_dn.append('Nan')
            else: # else, we know that it is a lifetime data, read the temprature, doping and dn from the title
                variable_type.append('X')
                temp, doping, dn = string.split('_')
                temp_list.append(temp)
                doping_level.append(doping)
                excess_dn.append(dn)
        return variable_type, temp_list, doping_level, excess_dn


    def pre_processor_dividX(self):
        """
        This function takes original data frame and multiply each lifetime data with (doping+dn)
        """
        # read off the doping and excess carrier concentration off the headings:
        variable_type, temp_list, doping_level, excess_dn = self.reader_heading()
        # for each column:
        for column_index in range(len(list(self.data.columns))):
            # if that column is a variable instead of y
            if variable_type[column_index] == 'X':
                # convert the readed text into numbers for dn and doping:
                doping_level[column_index] = float(doping_level[column_index].split('c')[0])
                excess_dn[column_index] = float(excess_dn[column_index].split('c')[0])
                T = float(temp_list[column_index].split('K')[0])
                # find the minority carrier concentration
                Tmodel=SRH(material="Si",temp = T, Nt = 1e12, nxc = excess_dn[column_index], Na = doping_level[column_index], Nd= 0, BGN_author = "Yan_2014fer")
                ni = Tmodel.nieff
                # calcualte the minority carrier concentration by ni**2=np*p0
                p0 = ni**2/doping_level[column_index]
                # multiply the lifetime in that column by (doping+dn)
                columnname = list(self.data.columns)[column_index]
                # print(self.data[columnname])
                self.data[columnname] = self.data[columnname]*(doping_level[column_index] + excess_dn[column_index] + p0)# /excess_dn[column_index]
                # print(self.data[columnname])

    def pre_processor_insert_dtal(self):
        """
        This function takes the original data frame from the object itself and add extra columns that is the difference between adjacent columns
        """
        # the plan for differentialtion is to add one column of zero on the left,one column of zero on the right
        # substract these two dataframes
        # delete the extra columns on the left and right
        # cascade original dataframe with this new one.

        # read the data from the object itself
        dataframe1 = self.data
        dataframe2 = self.data
        # only use the X part of the data:
        variable_type, temp_list, doping_level, excess_dn = self.reader_heading()
        dataframe1 = pd.DataFrame(dataframe1.iloc[:, np.array(variable_type)=='X'])
        dataframe2 = dataframe2.iloc[:, np.array(variable_type)=='X']
        # insert the zero columns:
        dataframe1['zeros']=0
        dataframe2.insert(0, 'zeros', np.zeros(np.shape(dataframe2.iloc[:,0])), allow_duplicates=True)
        # make the substraction
        difference = np.array(dataframe1) - np.array(dataframe2)
        # print(np.shape(dataframe1))
        # delete the first and last column:
        difference = difference[:, 1:-1] + np.finfo(np.float32).eps
        # cascade into new columns
        lifetimedata = self.data
        # make headings for the differences:
        columns = []
        for column in range(np.shape(difference)[1]):
            columns.append('differences' + str(column))
        difference = pd.DataFrame(difference, columns = columns)
        # print(difference)
        # print(np.shape(difference))
        combineddata = pd.concat([lifetimedata, np.abs(difference)], axis=1, join='inner')
        # update the processed data:
        self.data = combineddata


    def pre_processor(self):
        """
        This function do the data pre processing according to the task we wonna do

        input:
        the object itself (which is the lifetimedata) dataframe.
        singletask: a string of a column that we are doing y on

        output:
        X, y for maching learning purposes (before train test split and scaling)
        X: the lifetime data after taking log.
        y: the target volume for prediction.
        only work for single y output.
        """
        singletask = self.singletask # for now we make single taks same as task, in the future, we make task capable of doing multiple task.
        # define the columns to be deleted for ML purposes
        # delete_col = ['Name', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2', 'Mode', 'Label']
        delete_col = ['Name', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2']
        # drop these columns
        dfk = (pd.DataFrame(self.data)).drop(delete_col, axis=1)
        # define X and y based on the task we are doing.
        dfk = pd.DataFrame(dfk)
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in dfk.columns.tolist():
            if string[0].isdigit():
                select_X_list.append(string)
        X = dfk[select_X_list] # take the lifetime as X, delete any column that does not start with a number.
        # print(X.where(X==0))
        X = np.log10(X) # take the log of lifetime data.
        # in case we want to do some combination, pre-process the data based on the single task.
        if singletask == 'logk_1+logk_2':
            y = dfk['logk_1'] + dfk['logk_2']
        elif singletask == 'logk_1-logk_2':
            y = dfk['logk_1'] - dfk['logk_2']
        elif singletask == 'Et_eV_1+Et_eV_2':
            y = dfk['Et_eV_1'] + dfk['Et_eV_2']
        elif singletask == 'Et_eV_1_known_bandgap1':
            y = dfk['Et_eV_1']
            X['bandgap_1'] = dfk['bandgap_1']
        elif singletask == 'Et_eV_2_known_param1':
            y = dfk['Et_eV_2']
            X['Et_eV_1'] = dfk['Et_eV_1']
            X['logSn_1'] = pd.DataFrame(self.data)['logSn_1']
            X['logSp_1'] = pd.DataFrame(self.data)['logSp_1']
        elif singletask == 'Et_eV_1_known_predicted_bandgap_1':
            X['predicted_bandgap_1'] = dfk['predicted_bandgap_1']
            y = dfk['Et_eV_1']
        elif singletask == 'Et_eV_1-Et_eV_2':
            y = dfk['Et_eV_1'] - dfk['Et_eV_2']
        elif singletask == 'multi classification':
            # define the classification.
            y = dfk['bandgap_1'] + dfk['bandgap_2']
            # define y as a class.
        elif singletask == 'bandgap_2_known_1':
            # add the bandgap 1 and energy 1 into X.
            X['bandgap_1'] = dfk['bandgap_1']
            X['Et_eV_1'] = dfk['Et_eV_1']
            # define target as bandgap 2
            y = dfk['bandgap_2']
        elif singletask == 'Et_eV_2_known_Et_eV_2_plus':
            X['Et_eV_1'] = dfk['Et_eV_1']
            X['Et_eV_1+_Et_eV_2'] = dfk['Et_eV_1'] + dfk['Et_eV_2']
            y = dfk['Et_eV_2']
        elif singletask == 'Et_eV_2_known_Et_eV_1':
            y = dfk['Et_eV_2']
            X['Et_eV_1'] = dfk['Et_eV_1']
        elif singletask == 'Et_eV_1-Et_eV_2':
            y = dfk['Et_eV_1'] - dfk['Et_eV_2']
        # elif singletask == 'logSn_1':
        #     y = pd.DataFrame(self.data)['logSn_1']
        # elif singletask == 'logSp_1':
        #     y = pd.DataFrame(self.data)['logSp_1']
        elif singletask == 'multi_class_Et':
            y = pd.DataFrame(self.data)['bandgap_1'] + pd.DataFrame(self.data)['bandgap_2'] + 2*(pd.DataFrame(self.data)['bandgap_1']>pd.DataFrame(self.data)['bandgap_2'])
        elif singletask == 'whether 10':
            defectclass = pd.DataFrame(self.data)['bandgap_1'] + pd.DataFrame(self.data)['bandgap_2']
            y = defectclass == 1
        elif singletask == 'whether 11':
            defectclass = pd.DataFrame(self.data)['bandgap_1'] + pd.DataFrame(self.data)['bandgap_2']
            y = defectclass == 2
        else:
            y = pd.DataFrame(self.data)[singletask]
        # store the X and y to the object.
        # print(X)
        return X, y


    def bandgap_split(self):
        """
        This function aims to split the data into four sets based on the bandgap classification: 00 01 10 11
        Takes its own dataset and output 4 datasets of differnet bandgap configuration
        """
        # load the data from the object.
        df = self.data
        # split the dataframe:
        set11 = df[(np.array(df['bandgap_1']==1))*np.array(df['bandgap_2']==1)]
        set00 = df[(np.array(df['bandgap_1']==0))*np.array(df['bandgap_2']==0)]
        set10 = df[(np.array(df['bandgap_1']==1))*np.array(df['bandgap_2']==0)]
        set01 = df[(np.array(df['bandgap_1']==0))*np.array(df['bandgap_2']==1)]
        return set11, set10, set01, set00


    def pre_processor_insert_all_known(self):
        """
        This function will take each X column, then caluclate T, n, p, vn, vp, ni, Ei for each event
        """
        # Load the data from the object.
        dataframe = self.data
        # read the header of each data
        variable_type, temp_list, doping_level, excess_dn = self.reader_heading()
        # extract the X of the dataframe (features)
        features = pd.DataFrame(dataframe.iloc[:, np.array(variable_type)=='X'])
        # print(features)

        # itrate for each column: for each column, the T, n, p are the same
        # since vn and vp are only dependent on T.
        # ni is only dependent on T.
        # Ei is only dependent on T as well.
        extra_column_list = []
        lifetime_column_list = []
        column_name = []
        y_column_name = []
        # the original heading for the orginal lifetime data.
        headings = list(self.data.columns)
        y_columns = []
        for column_index in range(len(list(self.data.columns))):
            if variable_type[column_index] == 'y':
                # collect that into y columns.
                y_columns.append(self.data.iloc[:, column_index])
                y_heading = headings[column_index]
                y_column_name.append(y_heading)
            # if that column is a variable instead of y
            if variable_type[column_index] == 'X':
                # convert the readed text into numbers for dn and doping:
                doping_level[column_index] = float(doping_level[column_index].split('c')[0])
                excess_dn[column_index] = float(excess_dn[column_index].split('c')[0])
                T = float(temp_list[column_index].split('K')[0])
                T_heading = 'Temperature=' + str(T)
                # find the minority carrier concentration
                Tmodel=SRH(material="Si",temp = T, Nt = 1e12, nxc = excess_dn[column_index], Na = doping_level[column_index], Nd= 0, BGN_author = "Yan_2014fer")
                ni = float(Tmodel.nieff)
                ni_heading = 'ni=' + str(ni)
                # calcualte the minority carrier concentration by ni**2=np*p0
                p0 = ni**2/doping_level[column_index]
                # calculate the vn at this temperature.
                Vn = Tmodel.vel_th_e[0]
                Vp = Tmodel.vel_th_h
                Vn_heading = 'Vn=' + str(Vn)
                Vp_heading = 'Vp=' + str(Vp)
                # calcualte the carrier concentration
                majority_p = float(excess_dn[column_index] + doping_level[column_index])
                minority_n = p0 + excess_dn[column_index]
                majority_heading = 'Majority Carrier Concentration=' + str(majority_p)
                minority_heading = 'Minority Carrier Concentration=' + str(minority_n)
                # write up the extra columns.
                extra_column = np.array([T, majority_p, minority_n, Vn, Vp, ni])
                # print(extra_column)
                # # reshape the extra column the same dimension as the others.
                extra_column = np.zeros((np.shape(self.data)[0], len(extra_column))) + extra_column
                # print(np.shape(extra_column))
                # add this column after the current column.
                extra_column_list.append(extra_column)
                # print(np.shape(extra_column))

                # also collect the lifetime column.
                lifetime_column_list.append(self.data.iloc[:, column_index])
                # print(np.shape(self.data.iloc[:, column_index]))

                # in terms of the heading: firstly keep the original heading for the lifetime data.
                lifetime_heading = headings[column_index]
                column_name.append(lifetime_heading)
                # then append the heading for parameters.
                column_name.append(T_heading)
                column_name.append(majority_heading)
                column_name.append(minority_heading)
                column_name.append(Vn_heading)
                column_name.append(Vp_heading)
                column_name.append(ni_heading)

        # sanity check.
        # print(np.shape(extra_column_list)) [600, 8000, 6]
        # print(extra_column_list) [600, 8000]
        # print(np.shape(lifetime_column_list))
        # print(lifetime_column_list[0])# expect to see hte first column of lifeitme: met the expectation.
        X_array = np.empty((np.shape(extra_column_list)[1],))
        for event in range(np.shape(lifetime_column_list)[0]):
            # cascade the lifetime with its corresponding temperature and all other known information.
            # print(np.shape(extra_column_list[event])) (8000, 6) 8000 is the datasize, 6 is the number of parameter.
            # print(lifetime_column_list[event].ndim) # it is 1D array.
            # print(np.shape(lifetime_column_list[event])[0]) # 8000
            # lifetime_column_list[event] = np.reshape(lifetime_column_list[event], (np.shape((lifetime_column_list[event]))[0], -1))
            # event_array = np.concatenate((extra_column_list[event], lifetime_column_list[event]), axis=0)
            event_array = np.column_stack((lifetime_column_list[event], extra_column_list[event]))
            # collect the event array into the list.
            X_array = np.column_stack((X_array, event_array))
            # print(np.shape(event_array))
        X_array = X_array[:, 1:] # only have 3600 columns.
        # print(np.shape(X_array))
        # print(X_array[:, 0]) # expect to be found in the dataset as a lifetime. met the expectation.
        # print(X_array[:, -1]) # expect to be all the same: met the expectation.
        # put the heading for X.
        # print(np.shape(column_name))
        X_frame = pd.DataFrame(X_array, columns=column_name)
        # print(X_frame.head()) # expect to see the X_array with headings as a dataframe.
        # add X_frame with y frames.
        # print(np.shape(y_columns))
        y_frame = pd.DataFrame(np.transpose(np.array(y_columns)), columns=y_column_name)
        print(np.shape(y_frame)) # expect: 8000*19
        print(np.shape(X_frame))
        newdata = pd.concat([y_frame, X_frame], axis=1)
        print(np.shape(newdata)) # expect: 8000*(19+4200)
        # export it to csv to check the details.
        # newdata.to_csv('new_data.csv')


    def C1_C2_C3_C4_calculator(self, return_C=False, export=False):
        """
        This function only works if the data only have one temperature and one doping levels.

        This function calculate C1 C2 C3 C4 for each defect but instead of using n and p, it uses the intrinsic carrier concentration.

        return_C: whether or not to return C at the end.
        """
        # read off the temperature and doping and excess carrier concentration from the headings.
        variable_type, temp_list, doping_level, excess_dn = self.reader_heading()
        # read off and calcualte the vn, vp, n0, p0, ni, T, dn from the headings
        for column_index in range(len(list(self.data.columns))):
            # if that column is a variable instead of y
            if variable_type[column_index] == 'X':
                # convert the readed text into numbers for dn and doping:
                doping_level[column_index] = float(doping_level[column_index].split('c')[0])
                excess_dn[column_index] = float(excess_dn[column_index].split('c')[0])
                T = float(temp_list[column_index].split('K')[0])
                # find the minority carrier concentration
                Tmodel=SRH(material="Si",temp = T, Nt = 1e12, nxc = excess_dn[column_index], Na = doping_level[column_index], Nd= 0, BGN_author = "Yan_2014fer")
                ni = float(Tmodel.nieff)
                # calcualte the minority carrier concentration by ni**2=np*p0
                intrinsic_doping = doping_level[column_index]
                intrinsic_minority = ni**2/doping_level[column_index]
                # calculate the vn at this temperature.
                Vn = Tmodel.vel_th_e[0]
                Vp = Tmodel.vel_th_h
                # calcualte the carrier concentration
                break
        # for each defect, calculate C1 C2 C3 C4
        # create the list to collect them.
        C1d_list = []
        C2d_list = []
        C1n_list = []
        C2n_list = []
        # k1Vnn1_list = []
        # Vpdoping_list = []
        # Vpp1_list = []
        # k1Vnminority = []
        # iterate for each defect:
        # print(np.shape(self.data)) # expect 8000*feature numbers
        for row_index in range(np.shape(self.data)[0]):
            # read off the capcture cross sectional areas.
            k1 = self.data._get_value(row_index, 'k_1')
            k2 = self.data._get_value(row_index, 'k_2')
            Et1 = self.data._get_value(row_index, 'Et_eV_1')
            Et2 = self.data._get_value(row_index, 'Et_eV_2')
            Sn1 = self.data._get_value(row_index, 'Sn_cm2_1')
            Sn2 = self.data._get_value(row_index, 'Sn_cm2_2')
            Sp1 = self.data._get_value(row_index, 'Sp_cm2_1')
            Sp2 = self.data._get_value(row_index, 'Sp_cm2_2')
            # calculate n1 p1 and n2 p2
            n1 = ni*np.exp(Et1/(sc.k/sc.e)/T) # k here needs to be in eV/K
            p1 = ni**2/n1
            n2 = ni*np.exp(Et2/(sc.k/sc.e)/T)
            p2 = ni**2/n2
            # calcualte C1n C2n C1d and C2d
            C_2n = (Vp*p2 + k2*Vn*intrinsic_minority)/(k2*Vn*n2 + Vp*intrinsic_doping) # first term first energy
            C_2d = (Vn*Vp*Sn2*Sp2)/(Vn*n2*Sn2 + Vp*intrinsic_doping*Sp2) # second term second energy
            C_1n = (k1*Vn*n1 + Vp*intrinsic_doping)/(Vp*p1 + k1*Vn*intrinsic_minority) # first term first energy
            # C1_numerator = (k1*Vn*n1 + Vp*intrinsic_doping)
            # C1_denominator = (Vp*p1 + k1*Vn*intrinsic_minority)
            # k1Vnn1 = k1*Vn*n1
            # Vpdoping = Vp*intrinsic_doping
            # Vpp1 = Vp*p1
            # k1Vnminority = k1*Vn*intrinsic_minority
            C_1d = (Vn*Vp*Sp1*Sn1)/(Sn1*Vn*intrinsic_minority + Vp*p1*Sp1) # second term first energy
            # put them into the list
            C2n_list.append(C_2n)
            C2d_list.append(C_2d)
            C1n_list.append(C_1n)
            C1d_list.append(C_1d)
        # now insert the created C1, C2, C3, C4 into the original dataframe.
        newdata = self.data
        newdata['C1n'] = np.array(C1n_list)
        newdata['C2n'] = np.array(C2n_list)
        newdata['C1d'] = np.array(C1d_list)
        newdata['C2d'] = np.array(C2d_list)

        if export==True:
        # export as csv file to see what happens.
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            newdata.to_csv('Cdata' + str(current_time) + '.csv')

        # return the value if requried
        if return_C == True:
            return C1n_list, C2n_list, C1d_list, C2d_list


    def C1n_C2n_C1d_C2d_calculator(self, return_C=False, export=False, sanity_check=False, playmusic=False):
        """
        This function work for multiple T, and doping. But only works for p type material.

        This function calculate C1 C2 C3 C4 for each lifetiem points.

        return_C: whether or not to return C at the end.

        sanity_check: a boolean input that decide whether to use calculated C to calculate back tau to see if it matches the original data.
        """
        # read off the temperature and doping and excess carrier concentration from the headings.
        variable_type, temp_list, doping_level, excess_dn = self.reader_heading()

        # prepare emtply list to collect the C array for different T, doping and excess carrier concentration.
        C1d_array = []
        C2d_array = []
        C1n_array = []
        C2n_array = []
        T_title = []
        dn_title = []
        doping_title = []
        sanity_check_tau = []

        for column_index in range(len(list(self.data.columns))):
            # if that column is a variable instead of y
            if variable_type[column_index] == 'X':
                # convert the readed text into numbers for dn and doping:
                doping_level[column_index] = float(doping_level[column_index].split('c')[0])
                excess_dn[column_index] = float(excess_dn[column_index].split('c')[0])
                T = float(temp_list[column_index].split('K')[0])
                # find the minority carrier concentration
                Tmodel=SRH(material="Si",temp = T, Nt = 1e12, nxc = excess_dn[column_index], Na = doping_level[column_index], Nd= 0, BGN_author = "Yan_2014fer")
                ni = float(Tmodel.nieff)
                # calcualte the minority carrier concentration by ni**2=np*p0
                intrinsic_doping = doping_level[column_index]
                p0 = (0.5 * (np.abs(intrinsic_doping - 0) + np.sqrt((0 - intrinsic_doping)**2 + 4 * ni**2)))
                n0 = ni**2/p0
                # calculate the vn at this temperature.
                Vn = Tmodel.vel_th_e[0]
                Vp = Tmodel.vel_th_h
                # calcualte the excess carrier concentration
                dn = excess_dn[column_index]
                # calculate the carrier concentrations
                p = p0 + dn
                n = n0 + dn
                # write the heading for this column.
                T_title.append(str(T))
                dn_title.append(str(dn))
                doping_title.append(str(intrinsic_doping))
                # prepare the list to collect C for each defect column.
                C1d_list = []
                C2d_list = []
                C1n_list = []
                C2n_list = []
                tau_list = []
                # now iterate through the row (different T, doping and carrier concentration)
                # defect_counter = 0
                for row_index in range(np.shape(self.data)[0]):
                    # sanity check:
                    # defect_counter += 1
                    # if defect_counter > 10: continue
                    # read off the capcture cross sectional areas.
                    k1 = self.data._get_value(row_index, 'k_1')
                    k2 = self.data._get_value(row_index, 'k_2')
                    Et1 = self.data._get_value(row_index, 'Et_eV_1')
                    Et2 = self.data._get_value(row_index, 'Et_eV_2')
                    Sn1 = self.data._get_value(row_index, 'Sn_cm2_1')
                    Sn2 = self.data._get_value(row_index, 'Sn_cm2_2')
                    Sp1 = self.data._get_value(row_index, 'Sp_cm2_1')
                    Sp2 = self.data._get_value(row_index, 'Sp_cm2_2')
                    # calculate n1 p1 and n2 p2

                    # sanity check.
                    # print(sc.e)
                    # print(Et1)
                    # print(sc.k)
                    # print(T)
                    # print(ni)
                    # print(np.exp(float(float(Et1)*sc.e/sc.k/float(T))))

                    n1 = ni*np.exp(Et1*sc.e/(sc.k)/T) # k here needs to be in eV/K
                    p1 = ni*np.exp(-Et1*sc.e/sc.k/T)
                    n2 = ni*np.exp(Et2*sc.e/(sc.k)/T)
                    p2 = ni*np.exp(-Et2*sc.e/sc.k/T)
                    # calcualte C1n C2n C1d and C2d
                    C1n = (Sn1*n1*Vn + Sp1*Vp*p)/(Sp1*Vp*p1 + n*Sn1*Vn)
                    C2n = (Sn2*n*Vn + Sp2*Vp*p2)/(Sp2*Vp*p + n2*Sn2*Vn)
                    C1d = (Vn*Vp*Sn1*Sp1)/(Sp1*Vp*p1 + Sn1*Vn*n)
                    C2d = (Vn*Vp*Sn2*Sp2)/(Sp2*Vp*p + Sn2*Vn*n2)
                    # append the calculation to the list.
                    C2n_list.append(C2n)
                    C2d_list.append(C2d)
                    C1n_list.append(C1n)
                    C1d_list.append(C1d)
                    # sanity check to see if it maches the result.
                    Nt = 1e12
                    tau = (1 + C1n + C2n)/Nt/(n0 + p0 + dn)/(C1d + C2d)
                    tau_list.append(tau)

                    # sanity check
                    # print('p1 is ' + str(p1))
                    # print('n1 is ' + str(n1))
                    # print('p2 is ' + str(p2))
                    # print('n2 is ' + str(n2))
                    # print('Vn is ' + str(Vn))
                    # print('Vp is ' + str(Vp))
                    # print('Sn1 is ' + str(Sn1))
                    # print('Sn2 is ' + str(Sn2))
                    # print('Sp1 is ' + str(Sp1))
                    # print('Sp2 is ' + str(Sp2))
                    # print('Temperature is ' + str(T))
                    # print('Et1 is ' + str(Et1))
                    # print('Et2 is ' + str(Et2))
                    # print('ni is ' + str(ni))
                    # print('doping is ' + str(intrinsic_doping))
                    # print(str(tau))

                # collect the lists into the array.
                C2n_array.append(C2n_list)
                C2d_array.append(C2d_list)
                C1n_array.append(C1n_list)
                C1d_array.append(C1d_list)
                sanity_check_tau.append(tau_list)
                # print('Finish training column ' + str(column_index))


        # convert the list of list into array.
        C2n_array = np.transpose(np.array(C2n_array))
        C2d_array = np.transpose(np.array(C2d_array))
        C1n_array = np.transpose(np.array(C1n_array))
        C1d_array = np.transpose(np.array(C1d_array))
        sanity_check_tau = np.transpose(np.array(sanity_check_tau))

        # sanity check part:
        # if we export sanity check tau, we should see the same tau as the simulation data.
        if sanity_check == True:
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            pd.DataFrame(sanity_check_tau).to_csv('sanity check' + str(current_time) + '.csv')

        # print(np.shape(C1d_array)) # expect: (8000*3600)
        T_title = np.transpose(pd.DataFrame(np.transpose(np.array(T_title))))
        dn_title = np.transpose(pd.DataFrame(np.transpose(np.array(dn_title))))
        doping_title = np.transpose(pd.DataFrame(np.transpose(np.array(doping_title))))


        # before returning them, make them datagrame with proper headings instead of just arrays
        C2n_frame = pd.DataFrame(C2n_array)
        # add dn as extra row.
        C2n_frame = pd.concat([dn_title, C2n_frame], axis=0)
        # add temperature as an extra row.
        C2n_frame = pd.concat([T_title, C2n_frame], axis=0)
        # add doping as extra row.
        C2n_frame = pd.concat([doping_title, C2n_frame], axis=0)
        # do the same for other arrays.
        # for C2d
        C2d_frame = pd.DataFrame(C2d_array)
        C2d_frame = pd.concat([dn_title, C2d_frame], axis=0)
        C2d_frame = pd.concat([T_title, C2d_frame], axis=0)
        C2d_frame = pd.concat([doping_title, C2d_frame], axis=0)
        # for C1d
        C1d_frame = pd.DataFrame(C1d_array)
        C1d_frame = pd.concat([dn_title, C1d_frame], axis=0)
        C1d_frame = pd.concat([T_title, C1d_frame], axis=0)
        C1d_frame = pd.concat([doping_title, C1d_frame], axis=0)
        # for C1n.
        C1n_frame = pd.DataFrame(C1n_array)
        C1n_frame = pd.concat([dn_title, C1n_frame], axis=0)
        C1n_frame = pd.concat([T_title, C1n_frame], axis=0)
        C1n_frame = pd.concat([doping_title, C1n_frame], axis=0)

        # sanity check
        # print(C2n_frame.head()) # expect temperatures dopings dn in 3 different rows
        # print(np.shape(C2n_frame)) # expect 8003*3600
        # print(np.shape(T_title)) # expect 1*3600
        # print(np.shape(dn_title)) # expect 1*3600
        # print(np.shape(doping_title)) # expect 1*3600

        if export==True:
        # export as csv file to see what happens.
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            C2n_frame.to_csv('C2ndata' + str(current_time) + '.csv')
            C1n_frame.to_csv('C1ndata' + str(current_time) + '.csv')
            C2d_frame.to_csv('C2ddata' + str(current_time) + '.csv')
            C1d_frame.to_csv('C1ddata' + str(current_time) + '.csv')

        # play reminder playsound once the above steps are done:
        if playmusic == True:
            playsound(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\spongbob.mp3')

        # return the value if requried
        if return_C == True:
            return C2n_frame, C2d_frame, C1n_frame, C1d_frame


    def classfilter(self, filterclass=1):
        '''
        this function aims to filter out a cerain class of a two level defect from the dataset.
        '''
        defectclass = pd.DataFrame(self.data)['bandgap_1'] + pd.DataFrame(self.data)['bandgap_2']
        self.data = self.data[defectclass != filterclass]


    def dataset_integrator(self, path1, path2, path3, path4):
        '''
        input: the paths of the dataset to be integrated

        This function aims to:
        1. put the 3 different dataframe into one.
        2. shuffle them.
        '''

        # firstly: load the datasets:
        data1 = pd.read_csv(path1)
        data2 = pd.read_csv(path2)
        data3 = pd.read_csv(path3)
        data4 = pd.read_csv(path4)

        # integrate them together.
        integrated_data = pd.concat([data1, data2, data3, data4])

        # shuffle the integrated data.
        shuffled_data = integrated_data.sample(frac=1)

        # sanity check:
        # print(shuffled_data)

        # update the object data.
        self.data = shuffled_data


    def dataset_integrator2(self, path1, path2):
        '''
        input: the paths of the dataset to be integrated

        This function aims to:
        1. put the 3 different dataframe into one.
        2. shuffle them.
        '''

        # firstly: load the datasets:
        data1 = pd.read_csv(path1)
        data2 = pd.read_csv(path2)
        # data3 = pd.read_csv(path3)

        # integrate them together.
        integrated_data = pd.concat([data1, data2])

        # shuffle the integrated data.
        shuffled_data = integrated_data.sample(frac=1)

        # sanity check:
        # print(shuffled_data)

        # update the object data.
        self.data = shuffled_data

# %%-


# %%--- The functions for chain multi-output regressor chain.
    """
    note that multi-output only works for regression, not classification.
    The plan is to build the chain regressor instead of direct multi-output regressor
    The reason being is that: we have done the direct multi-output regression already using for loop
    """

    def sum_minus_Et1_chain(self, regression_order, plotall=False, return_pred=False):
        """
        This function perform chain regression for Et1->Et1+Et2.
        Then predict Et2 by Et1+Et2-Et1.

        input:
            1. regression_order: a list to spesify the order of which one to predict first.
            for example [0, 1] means predict the first column of y first then the second column.

            2. chain_name: a string input, see the pre processor for chain regression for more detail.

            3. plotall: a boolean input, if True, will plot real vs prediction for all models.

            4. return_pred: a boolean input, if True, will output the predictions of all models.


        output:
            r2matrix: a matrix of r2 score, the columns correspond to different task and the row correspond to different ML models.

            y_pred_list: dimension is (model, dataset size, Et1;Et2+Et1;regression chain;subtraction)

            y_test: dimension is (dataset size, Et1;Et2+Et1;Et2)

            r2_matrix: dimension is (model, Et1;Et1+Et2;Et2)
        """
        # do the normal train regression first.
        chain_name='Et1->Et1+Et2->Et2'
        X_train_scaled, X_test_scaled, y_train, y_test = self.preprocessor_chain_regression(chain_name=chain_name)
        # read the parameter setting from the object itself:
        model_names = self.reg_param['model_names']
        model_lists = self.reg_param['model_lists']
        gridsearchlist = self.reg_param['gridsearchlist']
        param_list  = self.reg_param['param_list']
        # iterate for each model:
        r2_matrix = []
        modelcount = 0
        y_pred_list = []
        for model in model_lists:
            # define the chained multioutput wrapper model
            wrapper = RegressorChain(model, order = regression_order)
            # fit the model on the whole dataset
            wrapper.fit(X_train_scaled, y_train)
            print('finish training chain regressor for ' + model_names[modelcount])
            # make the prediction:
            y_pred = wrapper.predict(X_test_scaled)
            # y_pred_list.append(y_pred)
            # y_pred should be a list and the element correspond to the y prediction based on the input order.
            # for single level case it is just Et or k
            # reorder the column of y_pred based on the input order:
            y_pred_ordered = y_pred
            # replace the last colume with the difference of the first two.
            # print(np.shape(y_pred_ordered))
            if plotall == True:
                print('The first 10 predicted Et1 are')
                print(y_pred_ordered[0:10, 0])
                print('The first 10 predicted Et1+Et2 are')
                print(y_pred_ordered[0:10, 1])
                print('The fist 10 Et2 prediction by machine learning are ')
                print(y_pred_ordered[0:10, -1])
            # add the colume for subtraction.
            # print(np.shape(y_pred_ordered))
            # print(np.shape(y_pred_ordered[0]))
            y_prediction = np.column_stack((y_pred_ordered, y_pred_ordered[:, 1]-y_pred_ordered[:, 0]))
            if plotall == True:
                print('The fist 10 Et2 prediction by subtraction are ')
                print(y_prediction[0:10, -1])
                print('The fist 10 real y value are ')
                print(np.array(y_test)[0:10, -1])
            y_pred_list.append(y_prediction)
            # print(np.shape(y_pred_ordered))
            # evaluate the matrix using r2 score:
            y_test = np.array(y_test)
            # prepare a list to collect the r2 values
            r2list = []
            # iterate for each variable: to calculate r2 score for Et1 Et2+Et1 Et2 by machine learning, Et2 by subtraction.
            # the y_test has 3 columes: Et1, Et2+Et1, Et2.
            for k in range(np.shape(y_test)[1]):
                r2 = (r2_score(y_test[:, k], y_pred_ordered[:, k]))
                r2list.append(r2)
                # find use k as index to call y_test title
                taskname = y_train.columns.tolist()[k]
                print('The R2 score for ' + str(taskname) + ' is ' + str(r2))
            # do the r2 score for subtraction prediction:
            r2 = r2_score(y_test[:, -1], y_prediction[:, -1])
            r2list.append(r2)
            print('The R2 score for subtraction method is ' + str(r2))

            r2_matrix.append(r2list)
            # r2_matrix = np.array(r2_matrix)
            tasknamelist = y_train.columns.tolist()

            # plot the behaviour of all models if requried
            if plotall == True:
                # iterate through each machine learning task.
                for k in range(np.shape(y_test)[1]):
                    plt.figure()
                    plt.scatter(y_test[:, k], y_pred_ordered[:, k], label='$R^2$=' + str(r2list[k]))
                    plt.xlabel('real value')
                    plt.ylabel('prediction')
                    plt.title('real vs prediction using model ' + str(model_names[modelcount]) + ' for ' + tasknamelist[k])
                    plt.legend()
                    plt.show()
                # plot the subtraction method behaviour:
                plt.figure()
                plt.scatter(y_test[:, -1], y_prediction[:, -1], label='$R^2$=' + str(r2list[-1]))
                plt.xlabel('real value')
                plt.ylabel('prediction')
                plt.title('real vs prediction using subtraction method for $E_{t2}$')
                plt.legend()
                plt.show()

            modelcount = modelcount + 1

        # plot the real vs predicted for all three machine learning tasks for the best trial of the last task.
        # find the model index for the best trial.
        # r2_matrix = np.array(r2_matrix)
        # modelindex = np.argwhere(r2_matrix[:, -2:-1] == np.max(r2_matrix[:, -2:-1]))[0][0]
        # print('the best R2 score is using ' + str(model_names[modelindex]))
        # # plot the prediction vs test for each tasks.
        # tasknamelist = y_train.columns.tolist()
        # best_y = y_pred_list[modelindex]
        # # iterate through each tasks.
        # for k in range(np.shape(y_test)[1]):
        #     plt.figure()
        #     plt.scatter(y_test[:, k], best_y[:, k], label='$R^2$=' + str(np.max(r2_matrix[modelindex, k])))
        #     plt.xlabel('real value')
        #     plt.ylabel('prediction')
        #     plt.title('real vs prediction using model ' + str(model_names[modelindex]) + ' for ' + tasknamelist[k])
        #     plt.legend()
        #     plt.show()
        # playsound('spongbob.mp3')
        r2_matrix = np.array(r2_matrix)
        if return_pred == True:
            return model_names, y_pred_list, y_test, r2_matrix
        return r2_matrix


    def chain_regression_once(self, regression_order, chain_name, plotall=False, return_pred=False):
        """
        This function perform chain regression on each parameter once.

        input:
            1. regression_order: a list to spesify the order of which one to predict first.
            for example [0, 1] means predict the first column of y first then the second column.

            2. chain_name: a string input, see the pre processor for chain regression for more detail.

            3. plotall: a boolean input, if True, will plot real vs prediction for all models.

            4. return_pred: a boolean input, if True, will output the predictions of all models.


        output:
            r2matrix: a matrix of r2 score, the columns correspond to different task and the row correspond to different ML models
        """
        X_train_scaled, X_test_scaled, y_train, y_test = self.preprocessor_chain_regression(chain_name=chain_name)
        # read the parameter setting from the object itself:
        model_names = self.reg_param['model_names']
        model_lists = self.reg_param['model_lists']
        gridsearchlist = self.reg_param['gridsearchlist']
        param_list  = self.reg_param['param_list']
        self.chain_name = chain_name
        # iterate for each model:
        r2_matrix = []
        mae_matrix = []
        modelcount = 0
        y_pred_list = []
        for model in model_lists:
            # define the chained multioutput wrapper model
            wrapper = RegressorChain(model, order = regression_order)
            # fit the model on the whole dataset
            wrapper.fit(X_train_scaled, y_train)
            print('finish training chain regressor for ' + model_names[modelcount])
            # make the prediction:
            y_pred = wrapper.predict(X_test_scaled)
            # y_pred_list.append(y_pred)
            # y_pred should be a list and the element correspond to the y prediction based on the input order.
            # for single level case it is just Et or k
            # reorder the column of y_pred based on the input order:
            y_pred_ordered = y_pred
            y_pred_list.append(y_pred_ordered)
            # y_pred_ordered = np.zeros_like(y_pred)
            # index2 = 0
            # for number in regression_order:
                # put the column into the right position.
            #     y_pred_ordered[:, number] = y_pred[:, index2]
                # update the index.
            #     index2 = index2 + 1
            # now the y_pred_ordered is the y_pred with the correct order as y_test.
            # notice that now y_test and y_pred are 2D matrix.
            # evaluate the matrix using r2 score:
            y_test = np.array(y_test)
            # prepare a list to collect the r2 values
            r2list = []
            # prepare a list to collect the mean absolute errors.
            mae_list = []
            # iterate for each variable:
            for k in range(np.shape(y_test)[1]):
                r2 = (r2_score(y_test[:, k], y_pred_ordered[:, k]))
                r2list.append(r2)
                # also append for mean absolute error
                mae = (mean_absolute_error(y_test[:, k], y_pred_ordered[:, k]))
                mae_list.append(mae)
                # find use k as index to call y_test title
                taskname = y_train.columns.tolist()[k]
                print('The R2 score for ' + str(taskname) + ' is ' + str(r2))
                print('The mean absolute error for ' + str(taskname) + 'is ' + str(mae))
            r2_matrix.append(r2list)
            mae_matrix.append(mae_list)
            tasknamelist = y_train.columns.tolist()

            # plot the behaviour of all models if requried
            if plotall == True:
                for k in range(np.shape(y_test)[1]):
                    plt.figure()
                    plt.scatter(y_test[:, k], y_pred_ordered[:, k], label='$R^2$=' + str(np.max(r2list[k])))
                    plt.xlabel('real value')
                    plt.ylabel('prediction')
                    plt.title('real vs prediction using model ' + str(model_names[modelcount]) + ' for ' + tasknamelist[k])
                    plt.legend()
                    plt.show()
            modelcount = modelcount + 1

        # plot the real vs predicted for all three machine learning tasks for the best trial of the last task.
        # find the model index for the best trial.
        r2_matrix = np.array(r2_matrix)
        mae_matrix = np.array(mae_matrix)
        if self.regression_matrix == 'R2':
            modelindex = np.argwhere(r2_matrix[:, -1] == np.max(r2_matrix[:, -1]))[0][0]
            print('the best R2 score is using ' + str(model_names[modelindex]))
        elif self.regression_matrix == 'Mean Absolute Error':
            modelindex = np.argwhere(mae_matrix[:, -1] == np.min(mae_matrix[:, -1]))[0][0]
            print('the lowest mean absolute error is using ' + str(model_names[modelindex]))
        # plot the prediction vs test for each tasks.
        tasknamelist = y_train.columns.tolist()
        best_y = y_pred_list[modelindex]
        for k in range(np.shape(y_test)[1]):
            plt.figure()
            plt.scatter(y_test[:, k], best_y[:, k], label='$R^2$=' + str((round(r2_matrix[modelindex, k], 3))) + '; Mean absolute error  is ' + str((round(mae_matrix[modelindex, k], 4))), marker='+')
            plt.xlabel('real value')
            plt.ylabel('prediction')
            plt.title('real vs prediction using model ' + str(model_names[modelindex]) + ' for ' + tasknamelist[k])
            plt.legend()
            plt.show()

        if return_pred == True:
            return model_names, y_pred_list, y_test, r2_matrix, mae_matrix
        return r2_matrix, mae_matrix


    def preprocessor_chain_regression(self, chain_name):
        """
        input:
        band: a string input being either plus or minus.
        output:
        X_train_scaled, X_test_scaled, y_train, y_test
        """
        # for now we make single taks same as task, in the future, we make task capable of doing multiple task.
        # define the columns to be deleted for ML purposes
        delete_col = ['Name', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2', 'Mode', 'Label']
        # drop these columns
        # drop these columns
        dfk = (pd.DataFrame(self.data)).drop(delete_col, axis=1)
        # if we are doing Et regression, we need to do them for above and below bandgap saperately
        # define X and y based on the task we are doing.
        dfk = pd.DataFrame(dfk)
        # create a list to select X columns: if the column string contains cm, then identify it as X.
        select_X_list = []
        for string in dfk.columns.tolist():
            if string.find('cm')!=-1:
                select_X_list.append(string)
        X = dfk[select_X_list] # take the lifetime as X, delete any column that does not start with a number.
        X = np.log(X) # take the log of lifetime data.
        # scale the data:
        for col in X.columns:
            # print(X[col])
            X[col] = MinMaxScaler().fit_transform(X[col].values.reshape(-1, 1))
        if chain_name == 'Et1->Et1+Et2->Et2':
            y = dfk[['Et_eV_1']]
            # also include the sum of energy level.
            y['Et_eV_1+Et_eV_2'] = dfk['Et_eV_1'] + dfk['Et_eV_2']
            y['Et_eV_2'] = dfk['Et_eV_2']
        elif chain_name == 'Et1->Et2':
            y = dfk[['Et_eV_1']]
            y['Et_eV_2'] = dfk['Et_eV_2']
        elif chain_name == 'logk1+logk2->logk1->logk2':
            y = np.array(dfk[['logk_1']]) + np.array(dfk[['logk_2']])
            y = pd.DataFrame(y)
            y.columns = ['logk_1+logk_2']
            y['logk_1'] = dfk['logk_1']
            y['logk_2'] = dfk['logk_2']
        elif chain_name == 'Et1->Et1+Et2->logk_1->logk_1+logk_2->Et2':
            y = dfk[['Et_eV_1']]
            # also include the sum of energy level.
            y['Et_eV_1+Et_eV_2'] = dfk['Et_eV_1'] + dfk['Et_eV_2']
            y['logk_1'] = dfk['logk_1']
            y['logk_1+logk_2'] = dfk['logk_1'] + dfk['logk_2']
            y['Et_eV_2'] = dfk['Et_eV_2']
        elif chain_name == 'Et1->Et1+Et2':
            y = dfk[['Et_eV_1']]
            # also include the sum of energy level.
            y['Et_eV_1+Et_eV_2'] = dfk['Et_eV_1'] + dfk['Et_eV_2']
        else:
            y = dfk[[chain_name[0]]]
            # in case the chain is input as a list.
            for columnname in chain_name[1:]:
                y[colume] = dfk[columnname]
        # print(y)
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.1)

        return X_train_scaled, X_test_scaled, y_train, y_test


    def repeat_chain_regressor(self, repeat_num, regression_order, chain_name='Et1->Et2', plotall=False, return_pred=False):
        """
        repeat the chain regressor for both plus and minus Et for multiple times
        input:
        repeat_num, number of repeat to do to

        output:
        y_pred_matrix: a matrix with dimension (repeatition number, model, taks number, number of sample)
        r2list: a matrix with dimension (repeition number, models, tasks)
        mae_list: a matrix with dimension (repetition number, models, tasks)
        """
        # prepare an empty list to collect different tasks r2 scores.
        r2list = []
        maelist = []
        y_pred_matrix = []
        # iterate for each repeatition
        for k in range(repeat_num):
            if return_pred == True:
                model_names, mae_matrix, y_pred_list, y_test, r2matrix = self.chain_regression_once(regression_order=regression_order, chain_name=chain_name, plotall=plotall, return_pred=return_pred)
                r2list.append(r2matrix)
                maelist.append(mae_matrix)
                y_pred_matrix.append(y_pred_list)
            else:
            # iterate for upper and lower bandgap
                r2matrix, mae_matrix = self.chain_regression_once(regression_order=regression_order, chain_name=chain_name, plotall=plotall)
                # we want to put the same task into the same table
                r2list.append(r2matrix)
                maelist.append(mae_matrix)
            print('finish repeatition ' + str(k+1))

        # make a boxplot for R2 scores.
        r2_frame = np.array(r2list)[:, :, -1] # the dimension of r2 frame is (repetition, models)
        r2_frame = pd.DataFrame(r2_frame, columns=self.reg_param['model_names'])
        r2_av = np.average(r2_frame, axis=0)
        r2_std = np.std(r2_frame, axis=0)
        labels = []
        # iterates through each
        for k in range(len(r2_av)):
            labels.append(str(r2_frame.columns[k] +' ('+ str(round(r2_av[k], 3)) + r'$\pm$' + str(round(r2_std[k], 3)) + ')'))
        # box plot the data.
        plt.figure()
        plt.boxplot(r2_frame, vert=False, labels=labels)
        plt.title('$R^2$ scores for ' + str(self.chain_name))
        # append the data label for the boxplot
        # for k in range(len(r2_av)):
        #     y = 8.5/(len(r2_av) + 1)*k + 0.5
        #     # x=0.99
        #     plt.text(x=0.98, y=y, s=str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)))
        plt.show()

        # make a boxplot for Mean Absolute Errors.
        mae_frame = np.array(maelist)[:, :, -1] # the dimension of mae frame is (repetition, models)
        mae_frame = pd.DataFrame(mae_frame, columns=self.reg_param['model_names'])
        mae_av = np.average(mae_frame, axis=0)
        mae_std = np.std(mae_frame, axis=0)
        labels = []
        # iterates through each
        for k in range(len(mae_av)):
            labels.append(str(mae_frame.columns[k] +' ('+ str(round(mae_av[k], 3)) + r'$\pm$' + str(round(mae_std[k], 4)) + ')'))
        # box plot the data.
        plt.figure()
        plt.boxplot(mae_frame, vert=False, labels=labels)
        plt.title('Mean Absolute error for ' + str(self.chain_name))
        # append the data label for the boxplot
        # for k in range(len(r2_av)):
        #     y = 8.5/(len(r2_av) + 1)*k + 0.5
        #     # x=0.99
        #     plt.text(x=0.98, y=y, s=str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)))
        plt.show()

        # print(np.shape(r2list))
        # play a nice reminder music after finishing
        # playsound('spongbob.mp3')
        if return_pred==True:
            return model_names, y_pred_matrix, y_test, r2list
        return r2list, mae_matrix


    def repeat_subtraction_method(self, repeat_num, regression_order, plotall=False, return_pred=False):
        """
        repeat the chain regressor for both plus and minus Et for multiple times
        input:
        repeat_num, number of repeat to do to

        output:
        y_pred_matrix: a matrix with dimension (repeatition number, model, taks number, number of sample)
        """
        model_names = self.reg_param['model_names']
        # prepare an empty list to collect different tasks r2 scores.
        r2list = []
        y_pred_matrix = []
        best_indexes = []
        best_scores = []
        y_test_matrix = []
        # iterate for each repeatition
        for k in range(repeat_num):
            model_names, y_pred_list, y_test, r2_matrix = self.sum_minus_Et1_chain(regression_order=regression_order, plotall=plotall, return_pred=return_pred)
            r2list.append(r2_matrix)
            y_test_matrix.append(y_test)
            y_pred_matrix.append(y_pred_list)
            print('finish repeatition ' + str(k+1))
            # find the best model.
            # the last two columns of r2_matrix are the Et2 score from machine learning and subtraction method respectively
            # print(r2_matrix)
            r2_Et2 = np.array(r2_matrix)[:, -2:-1]
            # find the highest score index.
            best_Et2_index = np.argwhere(r2_Et2 == np.max(r2_Et2))
            # store the best score and best indexes.
            best_indexes.append(best_Et2_index)
            best_scores.append(np.max(r2_Et2))
            # print(best_Et2_index[0])
            # extract the name of hte best model.
            bestmodel = model_names[best_Et2_index[0][0]]
            if best_Et2_index[0][1] == -1:
                bestmethod = 'subtraction'
            else:
                bestmethod = 'machine learning'
            print('The best R2 is ' + bestmethod + ' using ' + str(bestmodel) + str(np.max(r2_Et2)))

        # find the best trial of Et2 prediction.
        best_trial = np.argwhere(best_scores == np.max(best_scores))[0][0]
        # print(best_trial[0][0])
        # the first element of best index represent the model, the second element of best index represent whether it is subtraction or machine learning.
        best_index = best_indexes[best_trial]
        if best_index[0][1] == 1:
            bestmethod = 'subtraction'
        else:
            bestmethod = 'machine learning'
        title = model_names[best_index[0][0]] + bestmethod + ' method trial ' + str(best_trial)
        # find the prediction and test y
        # print(np.shape(y_pred_matrix))
        # print(np.shape(y_test_matrix))
        # the dimension of y_pred_matrix is (repetition, model, size of test set data, Et1;Et1+Et2;Et2 machine learning; Et2 subtraction)
        # the dimension of y_test_matrix is (repetition, datasize, Et1;Et2+Et1;Et2)
        # extract the prediction and test data for the best trial.
        best_prediction = np.array(y_pred_matrix)[best_trial, best_index[0][0], :, best_index[0][1] + 2]
        best_test = np.array(y_test_matrix)[best_trial, :, best_index[0][1] + 2]
        # print(best_index[0][1])
        # plot the best trial:
        plt.figure()
        plt.scatter(best_test, best_prediction, label='$R^{2}$' + '=' + str(np.max(best_scores)))
        plt.xlabel('prediction')
        plt.ylabel('real value')
        plt.title(str(title))
        plt.legend()
        plt.show()

        # print(np.shape(r2list))
        # the dimension of r2 list is (trial, model, machine learning tasks)
        # do the boxplot.
        # convert it into dataframe for box plot.
        r2list = np.array(r2list)
        for taskindex in range(2):
            r2_frame = pd.DataFrame(r2list[:, :, -taskindex-1], columns=model_names)
            r2_av = np.average(r2_frame, axis=0)
            r2_std = np.std(r2_frame, axis=0)
            labels = []
            # iterate through each model.
            for k in range(len(r2_av)):
                labels.append(str(r2_frame.columns[k] +' ('+ str(round(r2_av[k], 3)) + r'$\pm$' + str(round(r2_std[k], 3)) + ')'))
            # box plot the data.
            plt.figure()
            plt.boxplot(r2_frame, vert=False, labels=labels)
            if taskindex == 0:
                taskname = 'machine learning'
            else:
                taskname = 'subtraction'
            plt.title('$R^2$ scores for ' + '$E_{t2}$ ' + taskname + ' method')
            # append the data label for the boxplot
            # for k in range(len(r2_av)):
            #     y = 8.5/(len(r2_av) + 1)*k + 0.5
            #     # x=0.99
            #     plt.text(x=0.98, y=y, s=str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)))
            plt.show()
            # play a nice reminder music after finishing
        # playsound('spongbob.mp3')
        if return_pred==True:
            return model_names, y_pred_matrix, y_test, r2list
        return r2list
# %%-


# %%--- The functions for classification->regression chain
    def dataset_splitter(self, size=0.5):
        """
        When doing two different machine learning algarsim, to avoid data leakage, we need two different data frame for each task.
        this function will split the given dataframe into two dataframes
        input:
        size: is the size of the second set, 0.5 means divide half
        """
        set1, set2 = train_test_split(self.data, test_size=size)
        return set1, set2


    def apply_given_model(self, dataset, task, model, scaler):
        """
        This function aims to apply a given model on a dataset given the macine learning task.
        input:
        task: a string input that can be k,
        dataset: the dataset to perform the task on,
        model: the machine learning model to apply,
        scaler: the scaler used corresponding to this machine learning model.
        output:
        the prediction from model.
        """
        if task == 'bandgap_1':
            # do the same pre processing steps for bandgap 1 on the given dataset.
            self.singletask = task
            # define the columns to be deleted for ML purposes
            delete_col = ['Name', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2', 'Mode', 'Label']
            # drop these columns
            dfk = (pd.DataFrame(self.data)).drop(delete_col, axis=1)
            # define X and y based on the task we are doing.
            dfk = pd.DataFrame(dfk)
            # create a list to select X columns: if the column string contains cm, then identify it as X.
            select_X_list = []
            for string in dfk.columns.tolist():
                if string.find('cm')!=-1:
                    select_X_list.append(string)
            X = dfk[select_X_list] # take the lifetime as X, delete any column that does not start with a number.
            X = np.log(X) # take the log of lifetime data.
            y = dfk['Et_eV_1']
            # do the scaling: but make sure you only do scaling to the lifetime
            X_scaled = scaler.transform(X)
            # perform the prediction.
            y_pred = model.predict(X_scaled)
            return y_pred
# %%-


# %%--- Reminder functions to send reminders after finish ML trainings:
    def email_reminder(self):

        subject='ML finish training'
        body='ML of ' + str(self.singletask) + ' finished' + ' through the file ' + str(os.getcwd())
        to='z5183876@ad.unsw.edu.au'

        user = "sijinwang@yahoo.com"
        password = 'gdfkzhzhaokjivek'

        msg = EmailMessage()
        msg.set_content(body)
        msg['subject'] = subject
        msg['to'] = to
        msg['from'] = user



        server = smtplib.SMTP("smtp.mail.yahoo.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)

        server.quit()

    def playmusic(self):
        playsound(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\spongbob.mp3')
# %%-
