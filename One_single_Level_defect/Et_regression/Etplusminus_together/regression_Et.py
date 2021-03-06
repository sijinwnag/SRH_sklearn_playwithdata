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
# function definition

# define a function called 'outliers' which returns a list of index of outliers
# IQR = Q3-Q1
# boundary: +- 1.5*IQR
def outliers(df, ft, boundary):
    # input:
    # df: the data frame where we are looking for outliers
    # ft: the name of the feature that we are looking for outliers (string)
    # boundary: a number determine how wide is considered to be outliers, normally 1.5 or 3
    # output: a list of index of all the outliers.

    # start with calculating the quantiles
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)

    # calcualte the Interquartile range
    IQR = Q3 - Q1

    # define the upper and lower boundary for outliers.
    upper_bound = Q3 + boundary * IQR
    lower_bound = Q1 - boundary * IQR

    # collect the outliers
    out_list = df.index[(df[ft]<lower_bound) | (df[ft]>upper_bound)]

    return out_list


# define a function to find outliers for a list of futures
def outlier_ft_list(df, ft_list, inner_fence=True):
    # input:
    # df: the data frame where we are looking for outliers
    # ft: the name of the feature that we are looking for outliers (string)
    # inner_fence: if it is set to be true then the boundary is +-1.5*IQR, otherwise +-3*IQR
    # output: a list of index of all outliers for any feature in the ft_list.

    # decide whether use inner fence as outlier boundary or the outer fence.
    if inner_fence==True:
        boundary = 1.5
    else:
        boundary = 3

    out_list = []
    # find the outliers for each feature in a for loop
    for ft in ft_list:
        out_list.extend(outliers(df, ft, boundary))

    # remove the duplications
    out_list = list(dict.fromkeys(out_list))
    return out_list


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
    # here we try to keep Et for both above and below intrinsic fermi level to see what happens

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
    m_nn = MLPRegressor(hidden_layer_sizes=(100, 300, 300, 100))
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
    # n_repeat: the number of repetition needed
    # output:
    # r2_frame: a dataframe of r2 score.
    # meanabs_frame: a dataframe for mean absolute error.
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

# pre processing the data.
X_train_scaled, X_test_scaled, y_train, y_test = pre_processor(df)
# try shifting the Et by half band gap to approximate Et-Ev
Eg = 1.12
y_train = y_train + Eg/2
y_test = y_test + Eg/2

# train and evaluate the models.
# knn model.
mknn = KNeighborsRegressor()
param_knn = {'n_neighbors':range(1, 30)}
grid_knn = GridSearchCV(mknn, param_knn)
# fit the data
grid_knn.fit(X_train_scaled, y_train)
# evaluate the knn model
y_pred_knn = grid_knn.predict(X_test_scaled)
r2_knn = r2_score(y_test, y_pred_knn)
meanabs_knn = mean_absolute_error(y_test, y_pred_knn)
print('The R2 is: ' +str(r2_knn))
print('The mean absolute error is: ' + str(meanabs_knn))
plt.figure()
plt.scatter(y_test, y_pred_knn)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('knn predicted weekly sales ($)')
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
meanabs_ridge = mean_absolute_error(y_test, y_pred_ridge)
print('The R2 is: ' +str(r2_ridge))
print('The mean absolute error is: ' + str(r2_ridge))
plt.figure()
plt.scatter(y_test, y_pred_ridge)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('Ridge Linear Regression predicted weekly sales ($)')
plt.title('Ridge Linear Regression predicted vs real')
plt.show()

# try random Forest
m_rf = RandomForestRegressor()
# grid_rf = GridSearchCV(m_rf, param_rf)
# train the model with training dataset
m_rf.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_rf = m_rf.predict(X_test_scaled)
r2_rf = r2_score(y_test, y_pred_rf)
meanabs_rf = mean_absolute_error(y_test, y_pred_rf)
print('The R2 is: ' +str(r2_rf))
print('The mean absolute error is: ' + str(meanabs_rf))
plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('rf predicted weekly sales ($)')
plt.title('rf predicted vs real')
plt.show()
# export the data into excel for plotting with Et minus
E_plus_reg = pd.DataFrame(np.array([y_test, y_pred_rf]))
# E_plus_reg.to_csv('Et_reg_plus.csv')

# use neural Network
# rescale the y_train and y_test as well
y_train_scaled = y_train/np.max(y_train)
y_test_scaled = y_test/np.max(y_train)
m_nn = MLPRegressor(hidden_layer_sizes=(100, 300, 300, 100))
# param_nn = {'activation': ('identity', 'logistic', 'tanh', 'relu')}
# grid_nn = GridSearchCV(m_nn, param_nn)
m_nn.fit(X_train_scaled, y_train_scaled)
# m_nn.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_nn = m_nn.predict(X_test_scaled)
r2_nn = r2_score(y_test_scaled, y_pred_nn)
meanabs_nn = mean_absolute_error(y_test, y_pred_nn)
print('The R2 is: ' +str(r2_nn))
print('The mean absolute error is: ' + str(meanabs_nn))
plt.figure()
plt.scatter(y_test, y_pred_nn)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('nn predicted weekly sales ($)')
plt.title('nn predicted vs real')
plt.show()

# try more complext neural network: add more layers
# use neural Network
# rescale the y_train and y_test as well
y_train_scaled = y_train/np.max(y_train)
y_test_scaled = y_test/np.max(y_train)
m_nn = MLPRegressor(hidden_layer_sizes=(100, 300, 600, 300, 100))
# param_nn = {'activation': ('identity', 'logistic', 'tanh', 'relu')}
# grid_nn = GridSearchCV(m_nn, param_nn)
m_nn.fit(X_train_scaled, y_train_scaled)
# m_nn.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_nn = m_nn.predict(X_test_scaled)
r2_nn = r2_score(y_test_scaled, y_pred_nn)
meanabs_nn = mean_absolute_error(y_test, y_pred_nn)
print('The R2 is: ' +str(r2_nn))
print('The mean absolute error is: ' + str(meanabs_nn))
plt.figure()
plt.scatter(y_test, y_pred_nn)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('nn predicted weekly sales ($)')
plt.title('nn predicted vs real')
plt.show()

# try more complext neural network: increase layer sizes
# use neural Network
# rescale the y_train and y_test as well
y_train_scaled = y_train/np.max(y_train)
y_test_scaled = y_test/np.max(y_train)
m_nn = MLPRegressor(hidden_layer_sizes=(200, 600, 600, 200))
# param_nn = {'activation': ('identity', 'logistic', 'tanh', 'relu')}
# grid_nn = GridSearchCV(m_nn, param_nn)
m_nn.fit(X_train_scaled, y_train_scaled)
# m_nn.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_nn = m_nn.predict(X_test_scaled)
r2_nn = r2_score(y_test_scaled, y_pred_nn)
meanabs_nn = mean_absolute_error(y_test, y_pred_nn)
print('The R2 is: ' +str(r2_nn))
print('The mean absolute error is: ' + str(meanabs_nn))
plt.figure()
plt.scatter(y_test, y_pred_nn)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('nn predicted weekly sales ($)')
plt.title('nn predicted vs real')
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
meanabs_gb = mean_absolute_error(y_test, y_pred_gb)
print('The R2 is: ' +str(r2_gb))
print('The mean absolute error is: ' + str(meanabs_gb))
plt.figure()
plt.scatter(y_test, y_pred_gb)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('gb predicted weekly sales ($)')
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
meanabs_ab = mean_absolute_error(y_test, y_pred_ab)
print('The R2 is: ' +str(r2_ab))
print('The mean absolute error is: ' + str(meanabs_ab))
plt.figure()
plt.scatter(y_test, y_pred_ab)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('ab predicted weekly sales ($)')
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
meanabs_svr = mean_absolute_error(y_test, y_pred_svr)
print('The R2 is: ' +str(r2_svr))
print('The mean absolute error is: ' + str(meanabs_svr))
plt.figure()
plt.scatter(y_test, y_pred_svr)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('svr predicted weekly sales ($)')
plt.title('svr predicted vs real')
plt.show()
