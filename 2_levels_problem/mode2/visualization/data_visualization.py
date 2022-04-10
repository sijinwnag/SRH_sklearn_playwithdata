# plot a histagram for Et for level 1 and level 2:
# %%---import the library
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import sys
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\single_two_level_prob.csv')
# %%-



# %%--- histogram for Et1 and Et2
Et1 = df['Et_eV_1']
Et2 = df['Et_eV_2']
# plot the histogram:
plt.figure()
plt.hist(Et1)
plt.ylabel('number of defect')
plt.xlabel('Et')
plt.title('Histogram for Et1')

plt.figure()
plt.hist(Et2)
plt.ylabel('number of defect')
plt.xlabel('Et')
plt.title('Histogram for Et2')
# %%-

# %%--- Plot Et1 vs k1
Et1 = df['Et_eV_1']
k1 = df['k_1']
# plot the pairplot
plt.figure()
plt.scatter(Et1, np.log(k1))
plt.xlabel('$E_{t1}(eV)$')
plt.ylabel('$log(k_1)$')
plt.show()
# %%-

# %%--- find the feature importance for Et1.
# we know that random forest does the best job for regression, so only find the feature importance for random forest.
# we make X not only include the lifetime data, but also other y, so that we can see if we can use chain regression.
df1 = df
# np.shape(df1)
y = df1['Et_eV_1']
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Et_eV_1', 'Mode', 'Label']
X = df1.drop(delete_col, axis=1)
# now calculate the importance coefficient
model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
# train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# fit the data.
model.fit(X_train, y_train)
R2score = r2_score(y_test, model.predict(X_test))
print('The R2 score is ' + str(R2score))
importance = model.feature_importances_
# plot feature importance
# find the important features.
num = 5 # number of important feature to identify
importance2 = importance
X2 = X
for k in range(num):
    most_important = X2.columns[np.argmax(importance2)]
    print('The ' + str(k + 1) + ' most important feature is ' + str(most_important))
    # then delete the identified columns
    importance2 = np.delete(importance2, np.argmax(importance2))
    X2 = X2.drop([str(most_important)], axis=1)

# plot the importance of y features see what happens:
plt.figure()
plt.bar(X.columns[0:5], importance[0:5])
plt.ylabel('importance')
plt.title('importance of y on $E_t1$')
plt.show()

# print(np.shape(importance))
# plot the importance of X features see what happens:
plt.figure()
plt.plot(importance[5:])
plt.ylabel('importance')
plt.title('importance of lifetime on $E_{t1}$')
plt.show()
# %%-

# %%--- find the important features for Et2.
# we know that random forest does the best job for regression, so only find the feature importance for random forest.
# we make X not only include the lifetime data, but also other y, so that we can see if we can use chain regression.
df1 = df
y = df1['Et_eV_2']
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Et_eV_2', 'Mode']
def important_feature_identifier_rf(df1, targetname, delete_col, num=5):
    """
    The important columns for the given target column.

    input:
    df1: the dataframe we are given.
    targetname: the name of the dataframe column to find important features on
    delte_col: a list of string that contains the name of columns we want to delte from data frame to avoid data leakage
    num: the number of important columns to be identified
    """
    # define the target column:
    y = df1[targetname]
    # drop the redundant columns:
    X = df1.drop(delete_col, axis=1)
    # now calculate the importance coefficient
    model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
    # fit the data.
    model.fit(X, y)
    importance = model.feature_importances_
    # find the important features.
    importance2 = importance
    X2 = X
    for k in range(num):
        most_important = X2.columns[np.argmax(importance2)]
        print('The ' + str(k + 1) + ' most important feature is ' + str(most_important))
        # then delete the identified columns
        importance2 = np.delete(importance2, np.argmax(importance2))
        X2 = X2.drop([str(most_important)], axis=1)

important_feature_identifier_rf(df1, targetname='Et_eV_2', delete_col=delete_col, num=6)
# %%-

# %%--- find the important features for Et2, trial 2
# we know that random forest does the best job for regression, so only find the feature importance for random forest.
# we make X not only include the lifetime data, but also other y, so that we can see if we can use chain regression.
df1 = df
# np.shape(df1)
y = df1['Et_eV_2']
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Et_eV_2', 'Mode', 'Label']
X = df1.drop(delete_col, axis=1)
# now calculate the importance coefficient
model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
# train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# fit the data.
model.fit(X_train, y_train)
R2score = r2_score(y_test, model.predict(X_test))
print('The R2 score is ' + str(R2score))
importance = model.feature_importances_
# plot feature importance
# find the important features.
num = 5 # number of important feature to identify
importance2 = importance
X2 = X
for k in range(num):
    most_important = X2.columns[np.argmax(importance2)]
    print('The ' + str(k + 1) + ' most important feature is ' + str(most_important))
    # then delete the identified columns
    importance2 = np.delete(importance2, np.argmax(importance2))
    X2 = X2.drop([str(most_important)], axis=1)

# plot the importance of y features see what happens:
plt.figure()
plt.bar(X.columns[0:5], importance[0:5])
plt.ylabel('importance')
plt.title('importance of y on $E_t1$')
plt.show()

# print(np.shape(importance))
# plot the importance of X features see what happens:
plt.figure()
plt.plot(importance[5:])
plt.ylabel('importance')
plt.title('importance of lifetime on $E_{t1}$')
plt.show()

# %%-

# %%--- Find important features for logk1
# we know that random forest does the best job for regression, so only find the feature importance for random forest.
# we make X not only include the lifetime data, but also other y, so that we can see if we can use chain regression.
df1 = df
# np.shape(df1)
y = df1['logk_1']
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Mode', 'logk_1']
X = df1.drop(delete_col, axis=1)
# now calculate the importance coefficient
model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
# train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# fit the data.
model.fit(X_train, y_train)
R2score = r2_score(y_test, model.predict(X_test))
print('The R2 score is ' + str(R2score))
importance = model.feature_importances_
# plot feature importance
# find the important features.
num = 5 # number of important feature to identify
importance2 = importance
X2 = X
for k in range(num):
    most_important = X2.columns[np.argmax(importance2)]
    print('The ' + str(k + 1) + ' most important feature is ' + str(most_important))
    # then delete the identified columns
    importance2 = np.delete(importance2, np.argmax(importance2))
    X2 = X2.drop([str(most_important)], axis=1)

# plot the importance of y features see what happens:
plt.figure()
plt.bar(X.columns[0:5], importance[0:5])
plt.ylabel('importance')
plt.title('importance of y on $logk_1$')
plt.show()

# print(np.shape(importance))
# plot the importance of X features see what happens:
plt.figure()
plt.plot(importance[5:])
plt.ylabel('importance')
plt.title('importance of lifetime on $logk_1$')
plt.show()

# important_feature_identifier_rf(df1, targetname='logk_1', delete_col=delete_col, num=6)
# %%-

# %%-- Find importance for logk1 adding logk1+logk2 column.
# we know that random forest does the best job for regression, so only find the feature importance for random forest.
# we make X not only include the lifetime data, but also other y, so that we can see if we can use chain regression.
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\single_two_level_prob.csv')
df1 = df
df1.insert (3, "sum", df['logk_1']+df['logk_2'])
# np.shape(df1)
y = df1['logk_1']
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Mode', 'logk_1']
X = df1.drop(delete_col, axis=1)
# now calculate the importance coefficient
model = RandomForestRegressor(n_estimators=100, verbose =0, n_jobs=-1)
# train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# fit the data.
model.fit(X_train, y_train)
R2score = r2_score(y_test, model.predict(X_test))
print('The R2 score is ' + str(R2score))
importance = model.feature_importances_
# plot feature importance
# find the important features.
num = 5 # number of important feature to identify
importance2 = importance
X2 = X
for k in range(num):
    most_important = X2.columns[np.argmax(importance2)]
    print('The ' + str(k + 1) + ' most important feature is ' + str(most_important))
    # then delete the identified columns
    importance2 = np.delete(importance2, np.argmax(importance2))
    X2 = X2.drop([str(most_important)], axis=1)

# plot the importance of y features see what happens:
plt.figure()
plt.bar(X.columns[0:6], importance[0:6])
plt.ylabel('importance')
plt.title('importance of y on $logk_1$')
plt.show()

# print(np.shape(importance))
# plot the importance of X features see what happens:
plt.figure()
plt.plot(importance[6:])
plt.ylabel('importance')
plt.title('importance of lifetime on $logk_1$')
plt.show()
# %%-

# %%--- Find important features for logk2
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Mode', 'logk_2']
important_feature_identifier_rf(df1, targetname='logk_2', delete_col=delete_col, num=6)
# %%-

# %%--- Find important features for bandgap1
# we know that random forest does the best job for regression, so only find the feature importance for random forest.
# we make X not only include the lifetime data, but also other y, so that we can see if we can use chain regression.
df1 = df
# np.shape(df1)
y = df1['bandgap_1']
delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Mode', 'bandgap_1']
X = df1.drop(delete_col, axis=1)
# now calculate the importance coefficient
model = RandomForestClassifier(n_estimators=100, verbose =0, n_jobs=-1)
# train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# fit the data.
model.fit(X_train, y_train)
f1score = f1_score(y_test, model.predict(X_test), average='macro')
print('The f1 score is ' + str(f1score) + 'the confusion matrix is: ')
print(confusion_matrix(y_test, model.predict(X_test), normalize='all'))
importance = model.feature_importances_
# plot feature importance
# find the important features.
num = 5 # number of important feature to identify
importance2 = importance
X2 = X
for k in range(num):
    most_important = X2.columns[np.argmax(importance2)]
    print('The ' + str(k + 1) + ' most important feature is ' + str(most_important))
    # then delete the identified columns
    importance2 = np.delete(importance2, np.argmax(importance2))
    X2 = X2.drop([str(most_important)], axis=1)

# plot the importance of y features see what happens:
plt.figure()
plt.bar(X.columns[0:5], importance[0:5])
plt.ylabel('importance')
plt.title('importance of y on bandgap 1')
plt.show()

# print(np.shape(importance))
# plot the importance of X features see what happens:
plt.figure()
plt.plot(importance[5:])
plt.ylabel('importance')
plt.title('importance of lifetime on bandgap 1')
plt.show()

# %%-

# %%--- Find important features for bandgap2
targetname = 'bandgap_2'
delete_col = delete_col = ['Name', 'Sn_cm2_1', 'Sn_cm2_2', 'Sp_cm2_1', 'Sp_cm2_2', 'k_1', 'k_2', 'logSn_1', 'logSp_1', 'logSn_2', 'logSp_2', 'Mode']
delete_col.append(targetname)
important_feature_identifier_rf(df1, targetname=targetname, delete_col=delete_col, num=6)
# %%-
