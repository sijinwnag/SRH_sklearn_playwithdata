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


def classification_training(X_train_scaled, X_test_scaled, y_train, y_test):
    # input: X_train_scaled, X_test_scaled, y_train, y_test
    # output: a list of accuracy for each model

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
    print('finish knn model, the accuracy is: ' + str(knn_micro))

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
    print('finish svc model, the accuracy is: ' + str(svc_micro))

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
    print('finish decision tree model, the accuracy is: ' + str(knn_dt))

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
    print('finish random forest model, the accuracy is: ' + str(rf_micro))

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
    print('finish Gradient booster model, the accuracy is: ' + str(gb_micro))

    # Try Na??ve Bayes Classifiers
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
    print('finish Naive Bayes model, the accuracy is: ' + str(nb_micro))

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
    print('finish neural network model, the accuracy is: ' + str(nn_micro))

    # collect the f1 scores for each model into a list.
    f1_list = [knn_micro, svc_micro, dt_micro, rf_micro, gb_micro, nb_micro, nn_micro]

    return f1_list


def regression_repeat(df, n_repeat):
    # n_repeat: the number of repetition needed
    # output:
    # f1_frame: a dataframe of f1 score.
    # meanabs_frame: a dataframe for mean absolute error.
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


################################################################################
# firstly, load the data:
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# model training and evaluation
f1scores = regression_repeat(df, 5)

# create barchart to compare f1 scores.
avf1scores = np.average(f1scores, axis=0)
# avf1scores
# create a barchart
plt.figure()
models = ('KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector')
plt.barh(models, avf1scores)
plt.ylabel('f1 score')
plt.title(' average f1 score for Et regression below intrinsic fermi energy')
plt.show()

# export the data
f1scores.to_csv('Etminus_diffmodels.csv')

##################################################################################
# redefine the pre processor to use logX
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
    X = np.log(X)
    y = dfk['bandgap']
    # make the training size 0.9 and test size 0.1 (this is what was done by the paper)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # scale the data:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

##################################################################################
# repeat the step for logX and compare to original data
# model training and evaluation
f1scoreslog = regression_repeat(df, 5)

# create barchart to compare f1 scores.
avf1scoreslog = np.average(f1scores, axis=0)
# avf1scores
# create a barchart
plt.figure()
models = ('KNN', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Ada Boosting', 'Support Vector')
plt.barh(models, avf1scoreslog)
plt.ylabel('f1 score')
plt.title(' average f1 score for Et regression below intrinsic fermi energy')
plt.show()

# export the data
f1scoreslog.to_csv('Etminus_diffmodels.csv')
