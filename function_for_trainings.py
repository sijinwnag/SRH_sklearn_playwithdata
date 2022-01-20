"""
The function stroed in this files are:
regression_training: a function to do regression for single times with different algarisms
regression_repeat: a function to repeat regression for multiples times and plot the boxplot
classification_training: a function to do classification for single times with different algarisms
classification_repeat: a function to repeat classification for multiples times and plot the boxplot
"""

def regression_training(X_train_scaled, X_test_scaled, y_train, y_test, plot=False):

    """
    input:
        X_train_scaled, X_test_scaled, y_train, y_test
        plot: a boolean input, if True then it will plot real vs predicted for each model

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
    param_list  = [{'n_neighbors':range(1, 30)}, {'alpha': [0.01, 0.1, 1, 10]}, {'n_estimators': [10, 100]}, {'hidden_layer_sizes':((100, 300, 300, 100), (100, 300, 500, 300, 100), (200, 600, 600, 200))}, {'n_estimators':[10, 100]}, {'n_estimators':[10, 100]}, {'C': [0.1, 1, 10], 'epsilon': [1e-2, 0.1, 1]}]# a list of key parameters correspond to the models in the model_lists

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


    # this function will return all 2r scores and mean absolute errors
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
    plt.title('R2score for Sales regression models')
    plt.show()

    return r2_frame


def classification_training(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    input: X_train_scaled, X_test_scaled, y_train, y_test
    output: a list of accuracy for each model corresponding to 'KNN', 'SVC', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Naive Bayes', 'Neural Network'
    """
    # train the ML model: try knn with GridSearchCV varying the number of nearest neighbour
    mknn = KNeighborsClassifier()
    param_knn = {'n_neighbors':range(1, 30)}
    grid_knn = GridSearchCV(mknn, param_knn)
    # train the grid search with mknn.
    grid_knn.fit(X_train_scaled, y_train)
    # model evaluation for grid_knn
    # confusion matrix
    y_pred_knn = grid_knn.predict(X_test_scaled)
    # confusion_matrix(y_test, y_pred_knn)
    # macro f1 score.
    # knn_macro = f1_score(y_test, y_pred_knn, average='macro')
    # knn_macro
    # micro f1 score
    knn_micro = f1_score(y_test, y_pred_knn, average='micro')
    # accuracy
    # knn_accuracy = accuracy_score(y_test, y_pred_knn)
    # knn_accuracy

    # try using Kernalized Support Vector Machines
    msvc = SVC()
    param_svc = {'C': [0.1, 1, 10], 'kernel': ('linear', 'poly', 'rbf')}
    grid_svc = GridSearchCV(msvc, param_svc)
    # train the model using training Dataset
    grid_svc.fit(X_train_scaled, y_train)
    # model evaluation for SVC
    y_pred_svc = grid_svc.predict(X_test_scaled)
    # confusion matrix.
    # confusion_matrix(y_test, y_pred_svc)
    # macro f1 score.
    # svc_macro = f1_score(y_test, y_pred_svc, average='macro')
    # svc_macro
    # micro f1 score
    svc_micro = f1_score(y_test, y_pred_svc, average='micro')
    # svc_micro

    # Try decision tree
    mdt = DecisionTreeClassifier()
    param_dt = {'max_depth': [10, 100, 1e3]}
    grid_dt = GridSearchCV(mdt, param_dt)
    # train the model using training Dataset
    grid_dt.fit(X_train_scaled, y_train)
    y_pred_dt = grid_dt.predict(X_test_scaled)
    # model evaluation for decision tree.
    # confusion matrix.
    # confusion_matrix(y_test, y_pred_dt)
    # macro f1 score.
    # dt_macro = f1_score(y_test, y_pred_dt, average='macro')
    # dt_macro
    # micro f1 score
    dt_micro = f1_score(y_test, y_pred_dt, average='micro')
    # dt_micro
    # note that the behaviour of decision tree is identical to knn, the reason behind is unknown

    # Try random RandomForestClassifier
    mrf = RandomForestClassifier()
    param_rf = {'max_features':('auto', 'log2', '')}
    grid_rf = GridSearchCV(mrf, param_rf)
    # train the model with data.
    grid_rf.fit(X_train_scaled, y_train)
    # model evaluation for random RandomForestClassifier
    y_pred_rf = grid_rf.predict(X_test_scaled)
    # confusion matrix.
    # confusion_matrix(y_test, y_pred_rf)
    # macro f1 score.
    # rf_macro = f1_score(y_test, y_pred_rf, average='macro')
    # rf_macro
    # micro f1 score
    rf_micro = f1_score(y_test, y_pred_rf, average='micro')
    # rf_micro

    # Try Gradient boost
    mgb = GradientBoostingClassifier()
    param_gb = {'n_estimators':[100, 500, 1e3]}
    grid_gb = GridSearchCV(mgb, param_gb)
    # train the model using Dataset
    grid_gb.fit(X_train_scaled, y_train)
    # model evaluation for Gradient GradientBoostingClassifier
    y_pred_gb = grid_gb.predict(X_test_scaled)
    # confusion matrix.
    # confusion_matrix(y_test, y_pred_gb)
    # macro f1 score.
    # gb_macro = f1_score(y_test, y_pred_gb, average='macro')
    # gb_macro
    # micro f1 score
    gb_micro = f1_score(y_test, y_pred_gb, average='micro')
    # gb_micro

    # Try Naïve Bayes Classifiers
    mnb = GaussianNB()
    mnb.fit(X_train_scaled, y_train)
    # model evaluation for Naive Bayes Classifiers
    # model evaluation for Gradient GradientBoostingClassifier
    y_pred_nb = mnb.predict(X_test_scaled)
    # confusion matrix.
    # confusion_matrix(y_test, y_pred_nb)
    # macro f1 score.
    # nb_macro = f1_score(y_test, y_pred_nb, average='macro')
    # nb_macro
    # micro f1 score
    nb_micro = f1_score(y_test, y_pred_nb, average='micro')
    # nb_micro

    # Try Neural netwrok: one hot encoder for neural network classification
    # ohe = OneHotEncoder()
    # use the encoder to transform y
    # y_ohe_train = ohe.fit_transform(y_train).toarray()
    # y_ohe_train
    # y_ohe_test = ohe.fit_transform(y_test).toarray()
    # y_ohe_test
    m_nn = MLPClassifier(hidden_layer_sizes=(100, 300, 300, 300, 100))
    # param_nn = {'hidden_layer_sizes':((100, 300, 300, 300, 100), (200, 400, 400, 200), (100, 300, 300, 100))}
    # fit the data
    # grid_nn = GridSearchCV(m_nn, param_nn)
    m_nn.fit(X_train_scaled, y_train)
    # model evaluation for neural neural_network
    y_pred_nn = m_nn.predict(X_test_scaled)
    # y_pred_nn
    # encode the onehot encoded y back to original y
    # confusion matrix.
    # confusion_matrix(y_test, y_pred_nn)
    # macro f1 score.
    # nn_macro = f1_score(y_test, y_pred_nn, average='macro')
    # nn_macro
    # micro f1 score
    nn_micro = f1_score(y_test, y_pred_nn, average='micro')
    # nn_micro

    # collect the f1 scores for each model into a list.
    f1_list = [knn_micro, svc_micro, dt_micro, rf_micro, gb_micro, nb_micro, nn_micro]

    return f1_list


def classification_repeat(df, n_repeat):
    """
    input:
        df: the dataframe to work on
        n_repeat: the number of repetition needed
    output:
        f1_frame: a dataframe of f1 score.
        Also plotted a boxplot of f1 score for each model.
    """
    # set up counter to count the number of repetition
    counter = 0
    # create an emptly list to collect the f1 and mean absolute error values for each trials
    f1_frame = []
    meanabs_frame = []
    while counter < n_repeat:
        # update the counter
        counter = counter + 1
        # pro process the data:
        X_train_scaled, X_test_scaled, y_train, y_test = pre_processor(df)
        # train the different models and collect the f1 score.
        f1_frame.append(classification_training(X_train_scaled, X_test_scaled, y_train, y_test))
        # print the number of iteration finished after finishing each iteration
        print('finish iteration ' + str(counter))

    # now f1_frame is a list of list containing the values for each trial for each model.
    # convert it into dataframe for box plot.
    f1_frame = pd.DataFrame(f1_frame, columns=['KNN', 'SVC', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Naive Bayes', 'Neural Network'])
    # box plot the data.
    plt.figure()
    f1_frame.boxplot(vert=False)
    plt.title('f1score for Sales regression models')
    plt.show()

    return f1_frame
