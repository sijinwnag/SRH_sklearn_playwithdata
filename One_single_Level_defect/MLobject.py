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
# %%-


class MyMLdata:

    """
    MyMLdata is an object that is a panda dataframe containing the lifetime data.
    """

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
        regression_default_param = {
        'model_names': ['KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'], # a list of name for each model.
        'model_lists': [KNeighborsRegressor(), Ridge(), RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1), MLPRegressor((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive'), GradientBoostingRegressor(verbose=0,loss='ls',max_depth=10), AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='linear'), SVR(kernel='rbf',C=5,verbose=0, gamma="auto")],# a list of model improted from sklearn
        'gridsearchlist': [True, True, False, False, False, False, False], # each element in this list corspond to a particular model, if True, then we will do grid search while training the model, if False, we will not do Gridsearch for this model.
        'param_list': [{'n_neighbors':range(1, 30)}, {'alpha': [0.01, 0.1, 1, 10]}, {'n_estimators': [200, 100], 'verbose':0, 'n_jobs':-1}, {'hidden_layer_sizes':((100, 300, 300, 100), (100, 300, 500, 300, 100), (200, 600, 600, 200), (200, 600, 900, 600, 200)), 'alpha': [0.001], 'learning_rate':['adaptive']}, {'n_estimators':[200, 100]}, {'n_estimators':[50, 100]}, {'C': [0.1, 1, 10], 'epsilon': [1e-2, 0.1, 1]}]# a list of key parameters correspond to the models in the model_lists if we are going to do grid searching
        }
        classification_default_param = {
        'model_names': ['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'], # a list of name for each model.
        'model_lists': [KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, verbose =0,n_jobs=-1), GradientBoostingClassifier(verbose=0,loss='deviance'), AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10), GaussianNB(), MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=0,learning_rate='adaptive')],# a list of model improted from sklearn
        'gridsearchlist': [False, True, False, False, False, False, False, False],
        'param_list': [{'n_neighbors':range(1, 30)}, {'C': [0.1, 1, 10], 'kernel': ('linear', 'poly', 'rbf')},  {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]},{'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists
        }

        self.data = pd.read_csv(path)
        self.task = task
        self.repetition = repeat
        self.reg_param = regression_default_param
        self.cla_param = classification_default_param


    def pre_processor(self):
        """
        This function do the data pre processing according to the task we wonna do

        input:
        the object itself (which is the lifetimedata) dataframe.

        output:
        X, y for maching learning purposes (before train test split and scaling)
        """
        singletask = self.singletask # for now we make single taks same as task, in the future, we make task capable of doing multiple task.
        # define the columns to be deleted for ML purposes
        delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp']
        # drop these columns
        dfk = (pd.DataFrame(self.data)).drop(delete_col, axis=1)
        # if we are doing Et regression, we need to do them for above and below bandgap saperately
        if singletask == 'Et_plus':
            dfk = dfk[dfk['Et_eV']>0]
        elif singletask == 'Et_minus':
            dfk = dfk[dfk['Et_eV']<0]
        # define X and y based on the task we are doing.
        dfk = pd.DataFrame(dfk)
        X = np.log(dfk.drop(['logk', 'Et_eV', 'bandgap'], axis=1)) # takes the log of X to make it easier for ML
        if singletask == 'k':
            y = dfk['logk']
        elif singletask == 'Et_plus':
            y = dfk['Et_eV']
        elif singletask == 'Et_minus':
            y = dfk['Et_eV']
        elif singletask == 'bandgap':
            y = dfk['bandgap']

        # store the X and y to the object.
        return X, y


# %%--- The training and repeatition for regression task
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
            # scale the data:
            for col in X.columns:
                X[col] = MinMaxScaler().fit_transform(X[col].values.reshape(-1, 1))
            if self.singletask != 'Et_minus':
                y = y/np.abs(np.max(y))
            # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
            X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.1)
            # train the different models and collect the r2 score.
            if output_y_pred == True: # if we plan to collect the y predction
                r2score, y_prediction, y_test = self.regression_training(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train, y_test=y_test, plot=plot, output_y_pred=True)
                r2_frame.append(r2score)
                y_prediction_frame.append(y_prediction)
                y_test_frame.append(y_test)
            else: # when we do not need to collect the y prediction
                r2score = self.regression_training(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train, y_test=y_test, plot=plot)
                r2_frame.append(r2score)
            # print the number of iteration finished after finishing each iteration
            print('finish iteration ' + str(counter))
        # now r2_frame is a list of list containing the values for each trial for each model.
        # convert it into dataframe for box plot.
        r2_frame = pd.DataFrame(r2_frame, columns=['KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'])
        r2_av = np.average(r2_frame, axis=0)
        r2_std = np.std(r2_frame, axis=0)
        labels = []
        for k in range(len(r2_av)):
            labels.append(str(r2_frame.columns[k] +' ('+ str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)) + ')'))
        # box plot the data.
        plt.figure()
        plt.boxplot(r2_frame, vert=False, labels=labels)
        plt.title('R2 scores for ' + str(self.singletask))
        # append the data label for the boxplot
        # for k in range(len(r2_av)):
        #     y = 8.5/(len(r2_av) + 1)*k + 0.5
        #     # x=0.99
        #     plt.text(x=0.98, y=y, s=str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)))
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

        # use a for loop to train and evaluate each model:
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

            # scale the y back to original values
            y_pred_list.append(y_pred)
            y_test_list.append(y_test)
            # evaluate the model using R2 score:
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            # print the output
            print('finish training ' + name + ', the R2 score is ' + str(r2))
            # plot the real vs predicted graph if needed
            if plot==True:
                plt.figure()
                plt.scatter(y_test, y_pred)
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
            return r2_list, y_output, y_test
        # this function will return all 2r scores and mean absolute errors
        else:
            return r2_list
# %%-


# %%--- The training an repeatition for classification task
    def classification_training(self, X_train_scaled, X_test_scaled, y_train, y_test, display_confusion_matrix=False, output_y_pred=False):
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
                y_train = np.array(y_train)
                grid.fit(X_train_scaled, y_train)
                # use the trained model to predict the y
                y_pred = grid.predict(X_test_scaled)
            else:
                # just use the original model.
                model.fit(X_train_scaled, y_train)
                # predict with the original model using defalt settings
                y_pred = model.predict(X_test_scaled)

            # evaluate the model using micro f1 score (accuracy):
            f1 = f1_score(y_test, y_pred)
            f1_list.append(f1)
            y_pred_list.append(y_pred)
            y_test_list.append(y_test)
            # print the output
            print('finish training ' + name + ', the accuracy is ' + str(f1))
            # display the confusion matrix
            if display_confusion_matrix==True:
                print(confusion_matrix(y_test, y_pred, normalize='all'))

        if output_y_pred == True:
            return f1_list, y_pred_list, y_test_list
        else:
            return f1_list


    def classification_repeat(self, display_confusion_matrix=False, output_y_pred=False):
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
            f1_score, y_pred, y_test= self.classification_training(X_train_scaled, X_test_scaled, y_train, y_test, output_y_pred = output_y_pred)
            f1_frame.append(f1_score)
            if output_y_pred == True:
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
            labels.append(str(f1_frame.columns[k] +' ('+ str(round(f1_av[k], 3)) + '+-' + str(round(f1_std[k], 3)) + ')'))
        # box plot the data.
        plt.figure()
        plt.boxplot(f1_frame, vert=False, labels=labels)
        plt.title('f1score for classification')
        plt.show()

        if output_y_pred == False:
            return f1_frame
        else:
            return f1_frame, y_prediction_frame, y_test_frame
# %%-


    def perform_singletask_ML(self, plot_graphs=False):
        """
        This is the overall function to perform machine learning for a single task using the other functions

        Input: plot_graphs a boolean input, if true then the function will plot more detail graph after each training.

        What it does:
        1. identify what job is it doing.
        2. perform the maching learning process.
        3. print the evaluation
        """

        if self.task == 'k':
            # if the task is to do regression using k
            # apply the regression repeat function for k
            self.singletask = self.task
            r2_score_k, y_pred_k, y_test_k = self.regression_repeat(plot=plot_graphs, output_y_pred=True)
            # find the position which has the best R2 score.
            r2_score_k_output = r2_score_k
            r2_score_k = np.array(r2_score_k)
            max_position = np.argwhere(r2_score_k == np.max(r2_score_k))
            repeat_num = int(max_position[0][0])
            model_num = int(max_position[0][1])
            # plot the graph for real vs predicted
            plt.figure()
            plt.scatter(np.array(y_test_k)[repeat_num, :], np.array(y_pred_k)[repeat_num, :, model_num], label=('R2=' + str(round(np.max(r2_score_k), 3))))
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
                labels.append(str(r2_Et.columns[k] +' ('+ str(round(r2_av[k], 3)) + '+-' + str(round(r2_std[k], 3)) + ')'))
            # plot the r2 scores as a boxplot
            plt.figure()
            plt.boxplot(r2_Et, vert=False, labels=labels)
            plt.title('R2 scores for Et regression')
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
            plt.scatter(y_test_together[repeat_num, :], y_pred_together[repeat_num, :, model_num], label=('R2=' + str(round(np.max(r2_Et), 3))))
            plt.xlabel('real Et (eV)')
            plt.ylabel('predicted Et (eV)')
            plt.title('real vs predicted at trial ' + str(repeat_num + 1) + ' using method ' + str(self.reg_param['model_names'][model_num]))
            plt.legend()
            plt.show()

            return r2_Et_output

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
