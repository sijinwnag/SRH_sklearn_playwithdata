"""
The function stroed in this files are:
regression_training: a function to do regression for single times with different algarisms
regression_repeat: a function to repeat regression for multiples times and plot the boxplot
classification_training: a function to do classification for single times with different algarisms
classification_repeat: a function to repeat classification for multiples times and plot the boxplot
"""

def regression_training(X_train_scaled, X_test_scaled, y_train, y_test, plot=False, output_y_pred=False):

    """
    input:
        X_train_scaled, X_test_scaled, y_train, y_test
        plot: a boolean input, if True then it will plot real vs predicted for each model
        output_y_pred: a boolean input, if True then it will output the predicted y

    what it does: use the given data to train different regression algarisms

    output: a list of R2 scores for each model corresponding to 'KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'
    """
    # import libraries:
    import pandas as pd
    import numpy as np
    import seaborn as sn
    from sklearn.model_selection import train_test_split, GridSearchCV
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.linear_model import LinearRegression, Ridge
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    import sys

    # use a for loop to train and evaluate each model:
    model_names = ['KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'] # a list of name for each model.
    model_lists = [KNeighborsRegressor(), Ridge(), RandomForestRegressor(), MLPRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), SVR()]# a list of model improted from sklearn
    gridsearchlist = [True, True, False, True, False, False, True]
    param_list  = [{'n_neighbors':range(1, 30)}, {'alpha': [0.01, 0.1, 1, 10]}, {'n_estimators': [200, 100], 'verbose':2, 'n_jobs':-1}, {'hidden_layer_sizes':((100, 300, 300, 100), (100, 300, 500, 300, 100), (200, 600, 600, 200), (200, 600, 900, 600, 200)), 'alpha': [0.001], 'learning_rate':'adaptive'}, {'n_estimators':[200, 100]}, {'n_estimators':[50, 100]}, {'C': [0.1, 1, 10], 'epsilon': [1e-2, 0.1, 1]}]# a list of key parameters correspond to the models in the model_lists

    # prepare an emtply list to collect the predicted y
    y_pred_list = []
    # prepare an emtply list to collect r2 scores:
    r2_list = []
    # Prepare the y scaled data in case we need for neural network.
    y_train_scaled = y_train/np.max(y_train)
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
            grid.fit(X_train_scaled, y_train_scaled)
            # use the trained model to predict the y
            y_pred_scaled = grid.predict(X_test_scaled)
        else:
            # just use the original model.
            model.fit(X_train_scaled, y_train_scaled)
            # predict with the original model using defalt settings
            y_pred_scaled = model.predict(X_test_scaled)

        # scale the y back to original values
        y_pred = y_pred_scaled * np.max(y_train)
        y_pred_list.append(y_pred)
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
        return r2_list, y_output
    # this function will return all 2r scores and mean absolute errors
    else:
        return r2_list


def regression_repeat(X, y, n_repeat, plot=False):
    """
    input:
        X: the features.
        y: the target values.
        n_repeat: the number of times you want to repeat training test split and fitting the model.
        plot: if True, then predicted vs real will be plotted for each model after each training. if False, it will not plot anything.

    output:
        r2_frame: a dataframe, each row correspond to a trial and each column correspond to a model name.
        Also plot a boxplot of different model's R2 score
    """
    # import libraries:
    import pandas as pd
    import numpy as np
    import seaborn as sn
    from sklearn.model_selection import train_test_split, GridSearchCV
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.linear_model import LinearRegression, Ridge
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    import sys

    # set up counter to count the number of repetition
    counter = 0
    # create an emptly list to collect the r2 and mean absolute error values for each trials
    r2_frame = []
    meanabs_frame = []

    while counter < n_repeat:
        # update the counter
        counter = counter + 1
        # pro process the data:
        # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # scale the data:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # we must apply the scaling to the test set that we computed for the training set
        X_test_scaled = scaler.transform(X_test)
        # train the different models and collect the r2 score.
        r2_frame.append(regression_training(X_train_scaled, X_test_scaled, y_train, y_test, plot))
        # print the number of iteration finished after finishing each iteration
        print('finish iteration ' + str(counter))
    # now r2_frame is a list of list containing the values for each trial for each model.
    # convert it into dataframe for box plot.
    r2_frame = pd.DataFrame(r2_frame, columns=['KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'])
    # box plot the data.
    plt.figure()
    r2_frame.boxplot(vert=False)
    plt.title('R2 scores for different models')
    plt.show()

    return r2_frame


def classification_training(X_train_scaled, X_test_scaled, y_train, y_test, display_confusion_matrix=False):
    """
    This function is only capable for binary classification yet.
    input:
        X_train_scaled, X_test_scaled, y_train, y_test
        display_confusion_matrix: a boolean input if True then each time finish trianing a model will display its corresponding confusion matrix.

    output: a list of accuracy for each model corresponding to 'KNN', 'SVC', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Naive Bayes', 'Neural Network'
    """
    # import the libraries:
    import numpy as np
    import pandas as pd
    import seaborn as sn
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    import matplotlib.pyplot as plt

    model_names = ['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'] # a list of name for each model.
    model_lists = [KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), GaussianNB(), MLPClassifier()]# a list of model improted from sklearn
    gridsearchlist = [True, True, False, False, False, False, False, True]
    param_list  = [{'n_neighbors':range(1, 30)}, {'C': [0.1, 1, 10], 'kernel': ('linear', 'poly', 'rbf')},  {'max_depth': [10, 100, 1e3]}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]},{'n_estimators':[10, 100]}, {'var_smoothing':[1e-9, 1e-3]},{'hidden_layer_sizes':((100, 300, 500, 300, 100), (100, 300, 500, 500, 300, 100), (200, 600, 900, 600, 200))}]# a list of key parameters correspond to the models in the model_lists

    # prepare an emtply list to collect f1 scores:
    f1_list = []

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

        # evaluate the model using micro f1 score (accuracy):
        f1 = f1_score(y_test, y_pred)
        f1_list.append(f1)
        # print the output
        print('finish training ' + name + ', the accuracy is ' + str(f1))
        # display the confusion matrix
        if display_confusion_matrix==True:
            print(confusion_matrix(y_test, y_pred, normalize='all'))

    return f1_list


def classification_repeat(X, y, n_repeat, display_confusion_matrix=False):
    """
    input:
        df: the dataframe to work on
        n_repeat: the number of repetition needed
    output:
        f1_frame: a dataframe of f1 score.
        Also plotted a boxplot of f1 score for each model.
    """
    # import the libraries:
    import numpy as np
    import pandas as pd
    import seaborn as sn
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    import matplotlib.pyplot as plt

    # set up counter to count the number of repetition
    counter = 0
    # create an emptly list to collect the f1 and mean absolute error values for each trials
    f1_frame = []
    meanabs_frame = []
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
        f1_frame.append(classification_training(X_train_scaled, X_test_scaled, y_train, y_test, display_confusion_matrix))
        # print the number of iteration finished after finishing each iteration
        print('finish iteration ' + str(counter))

    # now f1_frame is a list of list containing the values for each trial for each model.
    # convert it into dataframe for box plot.
    f1_frame = pd.DataFrame(f1_frame, columns=['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network'])
    # box plot the data.
    plt.figure()
    f1_frame.boxplot(vert=False)
    plt.title('f1score for classification')
    plt.show()

    return f1_frame
