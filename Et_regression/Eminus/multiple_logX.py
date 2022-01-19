################################################################################
# import the library
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
################################################################################


def pre_processor(df):
    # input:
    # df: a dataframe of Walmart data csv.
    # output:
    # X_train_scaled
    # X-test_scaled.
    # y_train
    # y_test
    # what it does:
    # 1. delete all the columns that are not relevent
    # 2. define X and y
    # 3. train test split.
    # 4. Scale the X.

    # identify extract the useful columns
    # Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp logk bandgap are all y
    # here we only required to find k or logk (we do not know them when doing regression)
    # since logk has less range, we peak log k instead of k
    # Therefore, delete: Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp bandgap
    delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap', 'logk']
    dfk = df.drop(delete_col, axis=1)
    # we also need to make sure to delete all the rows that has Et > 0:
    dfk = dfk[dfk['Et_eV']<0]

    # train test split:
    X = dfk.drop(['Et_eV'], axis=1)
    y = dfk['Et_eV']
    # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # scale the data:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def regression_training(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    input: X_train_scaled, X_test_scaled, y_train, y_test

    what it does: use the given data to train different regression algarisms

    output: a list of R2 scores for each model corresponding to 'KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector'

    """

    # knn model.
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
    print('finish knn, the R2 score is: ' + str(r2_knn))
    plt.figure()
    plt.scatter(y_test, y_pred_knn)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('KNN predicted vs real')
    plt.show()

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
    print('finish ridge regression, the R2 score is: ' + str(r2_ridge))
    plt.figure()
    plt.scatter(y_test, y_pred_ridge)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('ridge predicted vs real')
    plt.show()


    # try random Forest
    m_rf = RandomForestRegressor()
    # grid_rf = GridSearchCV(m_rf, param_rf)
    # train the model with training dataset
    m_rf.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_rf = m_rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    # meanabs_rf = mean_absolute_error(y_test, y_pred_rf)
    print('finish random forest, the R2 score is: ' + str(r2_rf))
    plt.figure()
    plt.scatter(y_test, y_pred_rf)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('rf predicted vs real')
    plt.show()


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
    # meanabs_nn = mean_absolute_error(y_test, y_pred_nn)
    print('finish neural network, the R2 score is: ' + str(r2_nn))
    plt.figure()
    plt.scatter(y_test_scaled, y_pred_nn)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('NN predicted vs real')
    plt.show()


    # Try Gradient boosting Regression
    m_gb = GradientBoostingRegressor()
    # param_gb = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':[1, 5, 10]}
    # grid_gb = GridSearchCV(m_gb, param_gb)
    # train the model
    m_gb.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_gb = m_gb.predict(X_test_scaled)
    r2_gb = r2_score(y_test, y_pred_gb)
    # meanabs_gb = mean_absolute_error(y_test, y_pred_gb)
    print('finish gradient boosting, the R2 score is: ' + str(r2_gb))
    plt.figure()
    plt.scatter(y_test, y_pred_gb)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('gb predicted vs real')
    plt.show()


    # Try Adaptive boosting.
    m_ab = AdaBoostRegressor()
    # param_ab = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':[1, 5, 10]}
    # grid_ab = GridSearchCV(m_ab, param_ab)
    # train the model
    m_ab.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_ab = m_ab.predict(X_test_scaled)
    r2_ab = r2_score(y_test, y_pred_ab)
    # meanabs_ab = mean_absolute_error(y_test, y_pred_ab)
    print('finish adaboost Regression, the R2 score is: ' + str(r2_ab))
    plt.figure()
    plt.scatter(y_test, y_pred_ab)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('ab predicted vs real')
    plt.show()


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
    # meanabs_svr = mean_absolute_error(y_test, y_pred_svr)
    print('finish SVR, the R2 score is: ' + str(r2_svr))
    plt.figure()
    plt.scatter(y_test, y_pred_svr)
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('svr predicted vs real')
    plt.show()


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


################################################################################
# load the data.
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# train and evaluate the models.
r2scores = regression_repeat(df, 1)
# r2scores.to_csv('Etminus_diffmodels.csv')
# use r2scores to plot a barchart of average score for each model.
avr2scores = np.average(r2scores, axis=0)
# avr2scores
# create a barchart
plt.figure()
models = ('KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector')
plt.barh(models, avr2scores)
plt.ylabel('R2 score')
plt.title(' average R2 score for Et regression below intrinsic fermi energy')
plt.show()


################################################################################
# redefine the pro processor: this time we train and test with logX instead of X.
def pre_processor(df):
    # input:
    # df: a dataframe of Walmart data csv.
    # output:
    # X_train_scaled
    # X-test_scaled.
    # y_train
    # y_test
    # what it does:
    # 1. delete all the columns that are not relevent
    # 2. define X and y
    # 3. train test split.
    # 4. Scale the X.

    # identify extract the useful columns
    # Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp logk bandgap are all y
    # here we only required to find k or logk (we do not know them when doing regression)
    # since logk has less range, we peak log k instead of k
    # Therefore, delete: Name Et_eV Sn_cm2 Sp_cm2 k logSn logSp bandgap
    delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'bandgap', 'logk']
    dfk = df.drop(delete_col, axis=1)
    # we also need to make sure to delete all the rows that has Et > 0:
    dfk = dfk[dfk['Et_eV']<0]

    # train test split:
    X = dfk.drop(['Et_eV'], axis=1)
    X = np.log(X)
    y = dfk['Et_eV']
    # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # scale the data:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


###############################################################################
# repeat the process and compare the result
# load the data.
# df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')


# train and evaluate the models.
r2scoreslog = regression_repeat(df, 1)
# r2scores.to_csv('Etminus_diffmodels.csv')
# use r2scores to plot a barchart of average score for each model.
avr2scoreslog = np.average(r2scoreslog, axis=0)
# avr2scores
# create a barchart
plt.figure()
models = ('KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector')
plt.barh(models, avr2scoreslog)
plt.ylabel('R2 score')
plt.title(' average R2 score for Et regression below intrinsic fermi energy')
plt.show()
