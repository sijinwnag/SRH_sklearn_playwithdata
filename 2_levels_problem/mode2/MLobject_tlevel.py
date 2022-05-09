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
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
# %%-


class MyMLdata_2level:

    """
    MyMLdata is an object that is a panda dataframe containing the lifetime data.
    """
# %%--- Initialize the object
    def __init__(self, path, task, repeat):
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
        regression_default_param = {
        'model_names': ['KNN', 'Ridge Linear Regression'], # a list of name for each model.
        'model_lists': [KNeighborsRegressor(), Ridge()],# a list of model improted from sklearn
        'gridsearchlist': [True, True], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        'param_list': [{'n_neighbors':range(1, 30)}, {'alpha': [0.01, 0.1, 1, 10]}]
        }
        classification_default_param = {
        'model_names': ['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'], # a list of name for each model.
        'model_lists': [KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1), GradientBoostingClassifier(verbose=0,loss='deviance'), AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10), GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        'gridsearchlist': [False, False, False, False, False, False, False, False],
        'param_list': [{'n_neighbors':range(1, 30)}, {'C': [0.1, 1, 10], 'kernel': ('linear', 'poly', 'rbf')},  {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]},{'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        }
        # classification_default_param = {
        # 'model_names': ['KNN', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'], # a list of name for each model.
        # 'model_lists': [KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1), GradientBoostingClassifier(verbose=0,loss='deviance'), AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10), GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        # 'gridsearchlist': [False, False, False, False, False, False, False],
        # 'param_list': [{'n_neighbors':range(1, 30)}, {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]},{'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        # }

        self.data = pd.read_csv(path)
        self.singletask = task
        self.repetition = repeat
        self.reg_param = regression_default_param
        self.cla_param = classification_default_param
        self.regression_matrix = 'R2'
# %%-


# %%--- Regression machine learning tasks.
    def regression_repeat(self, plot=False, output_y_pred=False):
        # extract the X and y from previous step.
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
            X: the features.
            y: the target values.
            n_repeat: the number of times you want to repeat training test split and fitting the model.
            plot: if True, then predicted vs real will be plotted for each model after each training. if False, it will not plot anything.

        output:
            r2_frame: a dataframe, each row correspond to a trial and each column correspond to a model name.
            Also plot a boxplot of different model's R2 score
        """


        # set up counter to count the number of repetition
        counter = 0
        # create an emptly list to collect the r2 and mean absolute error values for each trials
        r2_frame = []
        meanabs_frame = []
        y_prediction_frame = []
        y_test_frame = []
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
            # we must apply the scaling to the test set that we computed for the training set
            X_test_scaled = scaler.transform(X_test)
            # train the different models and collect the r2 score.
            # if output_y_pred == True: # if we plan to collect the y predction
            r2score, mae_list, y_prediction, y_test = self.regression_training(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train, y_test=y_test, plot=plot, output_y_pred=True)
            r2_frame.append(r2score)
            meanabs_frame.append(mae_list) # the dimension is repeatition * different models.
            y_prediction_frame.append(y_prediction)
            y_test_frame.append(y_test)
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
        plt.figure()
        # print(np.shape(r2_frame))
        # print(np.shape(y_prediction_frame))
        # print(np.shape(y_test_frame))
        # print(np.shape(y_prediction_frame))
        plt.scatter(np.array(y_test_frame)[repeat_num], np.array(y_prediction_frame)[repeat_num, :, model_num], label=('$R^2$' + '=' + str(round(np.max(r2_score_k), 3))) + ('  Mean Absolue error' + '=' + str(round(np.min(mae_score_k), 3))), marker='+')
        plt.xlabel('real value')
        plt.ylabel('predicted value')
        plt.title('real vs predicted at trial ' + str(repeat_num + 1) + ' using method ' + str(self.reg_param['model_names'][model_num]))
        plt.legend()
        plt.show()

        if output_y_pred == False:
            return r2_frame
        else:
            return r2_frame, y_prediction_frame, y_test_frame


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
        # train everything in a for loop
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
            else:
                # just use the original model.
                model.fit(X_train_scaled, y_train)
                # predict with the original model using defalt settings
                y_pred = model.predict(X_test_scaled)

            # collect hte y values
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
            return r2_list, mae_list, y_output, y_test
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
        f1_frame = pd.DataFrame(f1_frame, columns=['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'])
        f1_av = np.average(f1_frame, axis=0)
        f1_std = np.std(f1_frame, axis=0)
        labels = []
        for k in range(len(f1_av)):
            labels.append(str(f1_frame.columns[k] +' ('+ str(round(f1_av[k], 3)) + r'$\pm$' + str(round(f1_std[k], 3)) + ')'))
        # box plot the data.
        plt.figure()
        plt.boxplot(f1_frame, vert=False, labels=labels)
        plt.title('$F_1$' + 'score for classification ' + str(self.singletask))
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


    def perform_alltasks_ML(self, plot_graphs=False):
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
        # in case we want to do some combination
        if singletask == 'logk_1+logk_2':
            y = dfk['logk_1'] + dfk['logk_2']
        elif singletask == 'logk_1-logk_2':
            y = dfk['logk_1'] - dfk['logk_2']
        elif singletask == 'Et_eV_1+Et_eV_2':
            y = dfk['Et_eV_1'] + dfk['Et_eV_2']
        elif singletask == 'Et_eV_1_known_bandgap1':
            y = dfk['Et_eV_1']
            X['bandgap_1'] = dfk['bandgap_1']
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
        elif singletask == 'Et_eV_1-Et_eV_2':
            y = dfk['Et_eV_1'] - dfk['Et_eV_2']
        else:
            y = dfk[singletask]
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
            # iterate for each variable:
            for k in range(np.shape(y_test)[1]):
                r2 = (r2_score(y_test[:, k], y_pred_ordered[:, k]))
                r2list.append(r2)
                # find use k as index to call y_test title
                taskname = y_train.columns.tolist()[k]
                print('The R2 score for ' + str(taskname) + ' is ' + str(r2))
            r2_matrix.append(r2list)
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
        modelindex = np.argwhere(r2_matrix[:, -1] == np.max(r2_matrix[:, -1]))[0][0]
        print('the best R2 score is using ' + str(model_names[modelindex]))
        # plot the prediction vs test for each tasks.
        tasknamelist = y_train.columns.tolist()
        best_y = y_pred_list[modelindex]
        for k in range(np.shape(y_test)[1]):
            plt.figure()
            plt.scatter(y_test[:, k], best_y[:, k], label='$R^2$=' + str(np.max(r2_matrix[modelindex, k])))
            plt.xlabel('real value')
            plt.ylabel('prediction')
            plt.title('real vs prediction using model ' + str(model_names[modelindex]) + ' for ' + tasknamelist[k])
            plt.legend()
            plt.show()

        if return_pred == True:
            return model_names, y_pred_list, y_test, r2_matrix
        return r2_matrix


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
        """
        # prepare an empty list to collect different tasks r2 scores.
        r2list = []
        y_pred_matrix = []
        # iterate for each repeatition
        for k in range(repeat_num):
            if return_pred == True:
                model_names, y_pred_list, y_test, r2matrix = self.chain_regression_once(regression_order=regression_order, chain_name=chain_name, plotall=plotall, return_pred=return_pred)
                r2list.append(r2matrix)
                y_pred_matrix.append(y_pred_list)
            else:
            # iterate for upper and lower bandgap
                r2matrix = self.chain_regression_once(regression_order=regression_order, chain_name=chain_name, plotall=plotall)
                # we want to put the same task into the same table
                r2list.append(r2matrix)
            print('finish repeatition ' + str(k+1))
        # play a nice reminder music after finishing
        # playsound('spongbob.mp3')
        if return_pred==True:
            return model_names, y_pred_matrix, y_test, r2list
        return r2list


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
