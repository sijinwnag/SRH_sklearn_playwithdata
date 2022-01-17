################################################################################
# import the library
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
    # delete all the defect information except bandgap
    delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'logk', 'Et_eV']
    dfk = df.drop(delete_col, axis=1)

    # train test split:
    X = dfk.drop(['bandgap'], axis=1)
    y = dfk['bandgap']
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
    y_pred_nn = grid_nn.predict(X_test_scaled)
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
# firstly, load the data:
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# go through data pre-processing:
X_train_scaled, X_test_scaled, y_train, y_test = pre_processor(df)

################################################################################
# model training and evaluation
# train the ML model: try knn with GridSearchCV varying the number of nearest neighbour
mknn = KNeighborsClassifier()
param_knn = {'n_neighbors':range(1, 10)}
grid_knn = GridSearchCV(mknn, param_knn)
# train the grid search with mknn.
grid_knn.fit(X_train_scaled, y_train)
# model evaluation for grid_knn
# confusion matrix
y_pred_knn = grid_knn.predict(X_test_scaled)
confusion_matrix(y_test, y_pred_knn)
# macro f1 score.
knn_macro = f1_score(y_test, y_pred_knn, average='macro')
knn_macro
# micro f1 score
knn_micro = f1_score(y_test, y_pred_knn, average='micro')
knn_micro
# accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_accuracy

# try using Kernalized Support Vector Machines
msvc = SVC()
param_svc = {'C': [0.1, 1, 10], 'kernel': ('linear', 'poly', 'rbf')}
grid_svc = GridSearchCV(msvc, param_svc)
# train the model using training Dataset
grid_svc.fit(X_train_scaled, y_train)
# model evaluation for SVC
y_pred_svc = grid_svc.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_svc)
# macro f1 score.
svc_macro = f1_score(y_test, y_pred_svc, average='macro')
svc_macro
# micro f1 score
svc_micro = f1_score(y_test, y_pred_svc, average='micro')
svc_micro

# Try decision tree
mdt = DecisionTreeClassifier()
param_dt = {'max_depth': [10, 100, 1e3]}
grid_dt = GridSearchCV(mdt, param_dt)
# train the model using training Dataset
grid_dt.fit(X_train_scaled, y_train)
y_pred_dt = grid_dt.predict(X_test_scaled)
# model evaluation for decision tree.
# confusion matrix.
confusion_matrix(y_test, y_pred_dt)
# macro f1 score.
dt_macro = f1_score(y_test, y_pred_dt, average='macro')
dt_macro
# micro f1 score
dt_micro = f1_score(y_test, y_pred_dt, average='micro')
dt_micro
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
confusion_matrix(y_test, y_pred_rf)
# macro f1 score.
rf_macro = f1_score(y_test, y_pred_rf, average='macro')
rf_macro
# micro f1 score
rf_micro = f1_score(y_test, y_pred_rf, average='micro')
rf_micro

# Try Gradient boost
mgb = GradientBoostingClassifier()
param_gb = {'n_estimators':[100, 500, 1e3]}
grid_gb = GridSearchCV(mgb, param_gb)
# train the model using Dataset
grid_gb.fit(X_train_scaled, y_train)
# model evaluation for Gradient GradientBoostingClassifier
y_pred_gb = grid_gb.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_gb)
# macro f1 score.
gb_macro = f1_score(y_test, y_pred_gb, average='macro')
gb_macro
# micro f1 score
gb_micro = f1_score(y_test, y_pred_gb, average='micro')
gb_micro

# Try Naïve Bayes Classifiers
mnb = GaussianNB()
mnb.fit(X_train_scaled, y_train)
# model evaluation for Naive Bayes Classifiers
# model evaluation for Gradient GradientBoostingClassifier
y_pred_nb = mnb.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_nb)
# macro f1 score.
nb_macro = f1_score(y_test, y_pred_nb, average='macro')
nb_macro
# micro f1 score
nb_micro = f1_score(y_test, y_pred_nb, average='micro')
nb_micro

# Try Neural netwrok: one hot encoder for neural network classification
# ohe = OneHotEncoder()
# use the encoder to transform y
# y_ohe_train = ohe.fit_transform(y_train).toarray()
# y_ohe_train
# y_ohe_test = ohe.fit_transform(y_test).toarray()
# y_ohe_test
m_nn = MLPClassifier(hidden_layer_sizes=(100, 300, 300, 300, 100))
# fit the data
m_nn.fit(X_train_scaled, y_train)
# model evaluation for neural neural_network
y_pred_nn = m_nn.predict(X_test_scaled)
# y_pred_nn
# encode the onehot encoded y back to original y
# confusion matrix.
confusion_matrix(y_test, y_pred_nn)
# macro f1 score.
nn_macro = f1_score(y_test, y_pred_nn, average='macro')
nn_macro
# micro f1 score
nn_micro = f1_score(y_test, y_pred_nn, average='micro')
nn_micro

# model comparison
# in terms of macro f1 score, each class has equal weight
macrof1 = plt.figure()
ax = macrof1.add_axes([10, 10, 1, 1])
model_selection = ['K Nearest Neighbors', 'Kernalized Support Vector Machines', 'Decision Trees', 'Random Forest', 'Gradient Boost', 'Naive Bayes', 'Neural network']
f1_macro_scores = [knn_macro, svc_macro, dt_macro, rf_macro, gb_macro, nb_macro, nn_macro]
ax.barh(model_selection,f1_macro_scores)
plt.title('Macro f1 scores')
plt.show()
# in terms of micro f1 score, each sample has equal weight
macrof1 = plt.figure()
ax = macrof1.add_axes([10, 10, 1, 1])
model_selection = ['K Nearest Neighbors', 'Kernalized Support Vector Machines', 'Decision Trees', 'Random Forest', 'Gradient Boost', 'Naive Bayes', 'Neural network']
f1_macro_scores = [knn_micro, svc_micro, dt_micro, rf_micro, gb_micro, nb_micro, nn_micro]
ax.barh(model_selection,f1_macro_scores)
plt.title('Micro f1 scores')
plt.show()
