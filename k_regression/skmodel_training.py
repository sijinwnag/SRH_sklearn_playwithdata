"""
The function stroed in this files are:
regression_training: a function to do regression for single times with different algarisms
regression_repeat: a function to repeat regression for multiples times and plot the boxplot
classification_training: a function to do classification for single times with different algarisms
classification_repeat: a function to repeat classification for multiples times and plot the boxplot
"""

def regression_training(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    input: X_train_scaled, X_test_scaled, y_train, y_test

    what it does: use the given data to train different regression algarisms

    output: a list of R2 scores for each model corresponding to 'KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'

    """

    # knn model.
    mknn = KNeighborsRegressor()
    param_knn = {'n_neighbors':range(1, 30)}
    grid_knn = GridSearchCV(mknn, param_knn)
    # fit the data
    grid_knn.fit(X_train_scaled, y_train)
    # evaluate the knn model
    y_pred_knn = grid_knn.predict(X_test_scaled)
    r2_knn = r2_score(y_test, y_pred_knn)
    # meanabs_knn = mean_absolute_error(y_test, y_pred_knn)


    # Linear Regression model.
    # use Linear Regression model now.
    m_ridge = Ridge()
    param_ridge = {'alpha': [0.01, 0.1, 1, 10]}
    # tune the model with parameters using grid search.
    grid_ridge = GridSearchCV(m_ridge, param_ridge)
    grid_ridge.fit(X_train_scaled, y_train)
    # evaluate the linear regression model
    y_pred_ridge = grid_ridge.predict(X_test_scaled)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    # meanabs_ridge = mean_absolute_error(y_test, y_pred_ridge)


    # try random Forest
    m_rf = RandomForestRegressor()
    # grid_rf = GridSearchCV(m_rf, param_rf)
    # train the model with training dataset
    m_rf.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_rf = m_rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    # meanabs_rf = mean_absolute_error(y_test, y_pred_rf)


    # use neural Network
    # rescale the y_train and y_test as well
    y_train_scaled = y_train/np.max(y_train)
    y_test_scaled = y_test/np.max(y_train)
    m_nn = MLPRegressor(hidden_layer_sizes = (100, 300, 300, 100))
    # param_nn = {'activation': ('identity', 'logistic', 'tanh', 'relu')}
    # grid_nn = GridSearchCV(m_nn, param_nn)
    m_nn.fit(X_train_scaled, y_train_scaled)
    # m_nn.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_nn = m_nn.predict(X_test_scaled)
    r2_nn = r2_score(y_test_scaled, y_pred_nn)
    meanabs_nn = mean_absolute_error(y_test, y_pred_nn)


    # Try Gradient boosting Regression
    m_gb = GradientBoostingRegressor()
    # param_gb = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':[1, 5, 10]}
    # grid_gb = GridSearchCV(m_gb, param_gb)
    # train the model
    m_gb.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_gb = m_gb.predict(X_test_scaled)
    r2_gb = r2_score(y_test, y_pred_gb)
    meanabs_gb = mean_absolute_error(y_test, y_pred_gb)


    # Try Adaptive boosting.
    m_ab = AdaBoostRegressor()
    # param_ab = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':[1, 5, 10]}
    # grid_ab = GridSearchCV(m_ab, param_ab)
    # train the model
    m_ab.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_ab = m_ab.predict(X_test_scaled)
    r2_ab = r2_score(y_test, y_pred_ab)
    meanabs_ab = mean_absolute_error(y_test, y_pred_ab)


    # Try Support vector regression
    m_svr = SVR()
    param_svr = {'C': [0.1, 1, 10], 'epsilon': [1e-2, 0.1, 1]}
    grid_svr = GridSearchCV(m_svr, param_svr)
    # train the model
    grid_svr = GridSearchCV(m_svr, param_svr)
    # train the model
    grid_svr.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_svr = grid_svr.predict(X_test_scaled)
    r2_svr = r2_score(y_test, y_pred_svr)
    meanabs_svr = mean_absolute_error(y_test, y_pred_svr)


    # this function will return all 2r scores and mean absolute errors
    return [r2_knn, r2_ridge, r2_rf, r2_nn, r2_gb, r2_ab, r2_svr]


def regression_repeat(df, n_repeat):
    """
    input:
        df: the dataframe you wanna apply regression on
        n_repeat: the number of times you want to repeat training test split and fitting the model.

    output:
        r2_frame: a dataframe, each row correspond to a trial and each column correspond to a model name.
        Also plot a boxplot of different model's R2 score
    """
    from multiple_logX import pre_processor
    # set up counter to count the number of repetition
    counter = 0
    # create an emptly list to collect the r2 and mean absolute error values for each trials
    r2_frame = []
    meanabs_frame = []
    while counter < n_repeat:
        # update the counter
        counter = counter + 1
        # pro process the data:
        X_train_scaled, X_test_scaled, y_train, y_test = pre_processor(df)
        # train the different models and collect the r2 score.
        r2_frame.append(regression_training(X_train_scaled, X_test_scaled, y_train, y_test))
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
