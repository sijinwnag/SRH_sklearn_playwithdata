# %%-- To do:
"""
generate result for set 11 p type 800k for five repetitions
"""
# %%-


# %%-- Imports
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
# uncomment the below line for dell laptop only
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
import scipy.constants as sc
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import sys
# import the function file from another folder:
# use this line if on hp laptop:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\DPML')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Savedir_example')
from MLobject_tlevel import *
# from dynamic_generation_regression import *
df1 = MyMLdata_2level(r"G:\study\thesis_data_storage\unordered\set11\p\set11_800k.csv", 'bandgap1',5)
# df1 = MyMLdata_2level(r"G:\study\thesis_data_storage\unordered\set11\p\set11_800k.csv", 'bandgap1',1)
df1.data.head()
# %%-

# %%-- different data engineering before training ML model.
# multiplying lifetime by (dn+p0+n0)
df1.pre_processor_dividX()
# %%-

# %%-- Single tasks.
# ['Et_eV_2', 'logSn_2', 'logSp_2', 'Et_eV_1', 'logSn_1', 'logSp_1']
for task in['Et_eV_1', 'logSn_1', 'logSp_1']:
    print(task)
    # refresh the dataset
    # df1 = MyMLdata_2level(r"G:\study\thesis_data_storage\unordered\set11\p\set11_p_800k.csv", 'bandgap1',5)
    df1.singletask = task
    r2_frame, y_prediction_frame, y_test_frame, best_model, scaler_return = df1.regression_repeat(output_y_pred=True)
    # reshape the test and prediction frame back to 2D:
    y_test_frame = pd.DataFrame(y_test_frame)
    y_prediction_frame = pd.DataFrame(y_prediction_frame)
    exportdata = pd.concat([y_test_frame, y_prediction_frame], axis=1)
    # export the validation data: name composed of: the singletask + the filename of the dataset.
    # df1.path
    filename = str(df1.singletask) + str(df1.path).split('\\')[-1]
    exportdata.to_csv(str(filename))
# %%-

# %%-- colour coding:
for column in ['logSn_2', 'logSp_2', 'Et_eV_2', 'logSn_1', 'logSp_1']:
    df1.colour_column = column
    df1.singletask = 'Et_eV_1'
    df1.colour_code_training()
# %%-

# %%-- Data leakage.
df1.singletask = 'Et_eV_2_known_Et_eV_2_plus'
r2scores = df1.regression_repeat()
# this makes the results better but has data leakage, R2 got about 0.999.

df1.singletask = 'Et_eV_2_known_param1'
r2scores = df1.regression_repeat()
df1.email_reminder()
# %%-

# %%-- Perform chain regression for energy levels.

# %%-- Just the chain.
df1.regression_matrix = 'Mean Absolute Error'
df1.regression_matrix = 'R2'
chain_scores = df1.repeat_chain_regressor(repeat_num=2, regression_order=None, chain_name = 'Et1->Et2')
chain_scores = df1.repeat_chain_regressor(repeat_num=2, regression_order=None, chain_name = 'Et1->Et1+Et2->Et2')
chain_scores = df1.repeat_chain_regressor(repeat_num=1, regression_order=None, plotall=True, chain_name = 'Et1->Sp1->Sn1->Sp2->Sn2->Et2')
# pd.DataFrame(np.array(chain_scores).reshape(35, 2)).to_csv(path_or_buf = r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\chainscore_two_steps.csv')
# %%-

# %%-- Chain and subtraction.
# the plan is to first predict Et1, then predict Et1+Et2, then predict Et2 by subtracting the prediction of sum by Et1 prediction.
# r2 = df1.sum_minus_Et1_chain(regression_order=None, plotall=True)
model_names, y_pred_matrix, y_test, r2list = df1.repeat_subtraction_method(repeat_num=1, regression_order=None, plotall=False, return_pred=True)
# %%-

# %%-

# %%-- Perform chain regression for k
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'logk1+logk2->logk1->logk2')
# %%-

# %%-- insert all known information as columns (failed)
df1.pre_processor_insert_all_known()
# %%-

# %%-- calculate C1 C2 C3 C4 as known for each defect.
# This equation only calcualte C while ignoring excess carrier concentration, and only works for one doping and one temperature.
# df1.C1_C2_C3_C4_calculator()
# this eqution works for lifetime data that vary both T and doping.
# df1.C1n_C2n_C1d_C2d_calculator(return_C=False, export=True, sanity_check=True, playmusic=True)
# %%-

# %%-- Data visualization

# %%-- General historam:
# histogram for C:
df1.C_visiaulization(variable='C1n/C2n')
df1.C_visiaulization()
# %%-

# %%-- Histogram for different T.
# plot for one temperauree:
# df1.C_visiaulization(task_name='histogram at T', T=150)

# plot for demoninator term.
for T in range(150, 401, 50):
    df1.C_visiaulization(task_name='histogram at T', T=T)
# plot for numeator term:
for T in range(150, 401, 50):
    df1.C_visiaulization(task_name='histogram at T', variable='C1n/C2n', T=T)
# why it seems like T does not change anything?
# %%-

# %%-- Histogram for different doping.
df1.C_visiaulization(task_name='histogram at doping', doping=1e14)
# # plot for demoninator term.
# for doping in [3e14, 7e14, 1e15, 3e15, 7e15, 1e16]:
#     df1.C_visiaulization(task_name='histogram at doping', doping=doping)
# # plot for numeator term:
# for doping in [3e14, 7e14, 1e15, 3e15, 7e15, 1e16]:
#     df1.C_visiaulization(task_name='histogram at doping', doping=doping, variable='C1n/C2n')
# # why it seems like doping does not change anything either.
# %%-

# %%-- Visialize individual parameters.
df1.C_visiaulization(variable='C1n')
df1.C_visiaulization(variable='C2n')
df1.C_visiaulization(variable='C1d')
df1.C_visiaulization(variable='C2d')
df1.C_visiaulization(task_name='C histogram compare')
df1.C
# %%-

# %%-- T vs C:
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with T')
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with T')
# %%-

# %%-- Doping vs C:
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with doping')
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with doping')
# %%-

# %%-- dn vs C:
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with dn')
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with dn')
# %%-

# %%-- E_diff vs C:
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with Et1-Et2')
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with Et1-Et2')
# %%-

# %%-- data importance visualization
df1.feature_importance_visualisation('Et_eV_2')
# %%-

# %%-- histogram for defect charge state population:
C2n_frame, C2d_frame, C1n_frame, C1d_frame = df1.C1n_C2n_C1d_C2d_calculator(return_C=True, export=False, sanity_check=False, playmusic=False)
# now we want to compare the results for C1n and C2n:
C1n_framedata = C1n_frame.iloc[3:, :]
C1n_av = np.mean(np.array(C1n_framedata), axis=1).reshape(np.shape(C1n_av)[0], 1).astype(float)
C1n_avlog = np.log10(np.array(C1n_av))
# extract teh C2n as well.
C2n_framedata = C2n_frame.iloc[3:, :]
# C2n_framedata
C2n_av = np.mean(np.array(C2n_framedata), axis=1).reshape(np.shape(C1n_av)[0], 1).astype(float)
C2n_avlog = np.log10(np.array(C2n_av))
C1n_av2 = np.mean(C1n_av)
C2n_av2 = np.mean(C2n_av)
print(C1n_av2/C2n_av2)
# %%--
# plot the histogram comparison:
labels=['most positively charge / middle charge', 'most negatively charge / middle charge']
plt.figure()
plt.boxplot(np.concatenate([C1n_avlog, C2n_avlog], axis=1), vert=False, labels=labels, showfliers=True)
# plt.title('Mean absolute error scores for ' + str(self.singletask))
plt.xlabel('log10 of the value')
plt.show()
# %%-
# %%-
# %%-

# %%-- test the first of dynamic generation method: use ML object.
df1 = MyMLdata_2level(r"G:\study\thesis_data_storage\set11\set11_80000.csv", 'bandgap1',1)

# predict Et1:
df1.singletask='logSn_1'
r2_frame, y_prediction_frame, y_test_frame, best_model, scaler_return = df1.regression_repeat(output_y_pred=True)
# now we have new lifetiem data from another file: load the lifetime data:
validationdata = pd.read_csv(r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\lifetimedata\point3\set11_50_1.csv")
# validationdata = pd.read_csv(r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_1.csv")
# extract the lifetime data:
select_X_list = []
validationsetX = validationdata
for string in validationdata.columns.tolist():
    if string[0].isdigit():
        # take the log of the data.
        select_X_list.append(string)
# extract the lifetime data.
validationsetX = validationdata[select_X_list]
# print(validationsetX)
# take the log:
validationsetX = np.log10(validationsetX)
# print(validationsetX)
# go through the scaler.
validationsetX = scaler_return.transform(validationsetX)
# print(validationsetX)
# Model to predict:
y_pred = best_model.predict(validationsetX)
print(y_pred)
df1.email_reminder()
# %%-

# %%-- test the second of dynamic generation method: use ML object.
# assume at this step the data generation for second step is done:
df1 = MyMLdata_2level(r"G:\study\thesis_data_storage\set11\ddgm\point1\dataset2_withAuger.csv", 'bandgap1',1)
df1.singletask = 'logSp_2'

# try to do without pre processor or manually.
r2_frame, y_prediction_frame, y_test_frame, best_model, scaler_return = df1.regression_repeat(output_y_pred=True)

# now we have new lifetiem data from another file: load the lifetime data:
validationdata = pd.read_csv(r"G:\study\thesis_data_storage\set11\ddgm\point1\set11_50_1.csv")
# validationdata = pd.read_csv(r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_1.csv")
# extract the lifetime data:
select_X_list = []
validationsetX = validationdata
for string in validationdata.columns.tolist():
    if string[0].isdigit():
        # take the log of the data.
        select_X_list.append(string)
# extract the lifetime data.
validationsetX = validationdata[select_X_list]
print(validationsetX)
# print(validationsetX)
# take the log:
validationsetX = np.log10(validationsetX)
print(validationsetX)
# print(validationsetX)
# go through the scaler.
validationsetX = scaler_return.transform(validationsetX)
print(validationsetX)
# print(validationsetX)
# Model to predict:
y_pred = best_model.predict(validationsetX)
print(y_pred)
# %%-

# %%-- test the idea of dynamic genration method: from scrach, no scaler nor log10.
# assume at this step the data generation for second step is done: load the data:
trainingset = pd.read_csv(r"G:\study\thesis_data_storage\set11\ddgm\point1\dataset2_withAuger.csv")

# extract the lifeitme training data.
select_X_list = []
for string in trainingset.columns.tolist():
    if string[0].isdigit():
        # take the log of the data.
        select_X_list.append(string)
trainingX = trainingset[select_X_list]

# define the ML model.
model = RandomForestRegressor(n_estimators=150)

# extract the target value:
y = trainingset['logSp_2']

# traing the model.
model.fit(trainingX, y)

# now we have new lifetiem data from another file: load the lifetime data:
validationdata = pd.read_csv(r"G:\study\thesis_data_storage\set11\ddgm\point1\set11_50_1.csv")

# extract the lifetime data.
validationsetX = validationdata[select_X_list]

# Model to predict:
y_pred = model.predict(validationsetX)
print(y_pred)
# %%-

# %%-- test the idea of dynamic genration method: from scrach, but with log10 and scalers. (using predicted Et1, Sn1, Sp1)
# assume at this step the data generation for second step is done: load the data:
trainingset = pd.read_csv(r"G:\study\thesis_data_storage\set11\ddgm\point1\dataset2_withAuger.csv")

# extract the lifeitme training data.
select_X_list = []
for string in trainingset.columns.tolist():
    if string[0].isdigit():
        # take the log of the data.
        select_X_list.append(string)
trainingX = trainingset[select_X_list]

# take log10 of the data.
trainingX = np.log10(trainingX)

# apply a scaler on the data.
scaler = MinMaxScaler()
scaler.fit_transform(trainingX)

# define the ML model.
model = RandomForestRegressor(n_estimators=150)

# extract the target value:
y = trainingset['logSp_2']

# traing the model.
model.fit(trainingX, y)

# now we have new lifetiem data from another file: load the lifetime data:
validationdata = pd.read_csv(r"G:\study\thesis_data_storage\set11\ddgm\point1\set11_50_1.csv")

# extract the lifetime data.
validationsetX = validationdata[select_X_list]

# take the log for validation data.
validationsetX = np.log10(validationsetX)
# go through the scaler.
validtionsetX = scaler.transform(validationsetX)

# Model to predict:
y_pred = model.predict(validationsetX)
print(y_pred)
# %%-

# %%-- test the dynamic regression object.
# training_path = r"G:\study\thesis_data_storage\set11\set11_80000.csv"
# validation_path = r"G:\study\thesis_data_storage\set11\set11_50.csv"
validation_path = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_1.csv"
training_path = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_80000.csv"
dy = Dynamic_regression(training_path=training_path, validation_path = validation_path, noise_factor=0, simulate_size=8000, n_repeat=1)
dy.t_step_train_predict()
# export the data
# pd.DataFrame(dy.y_predictions_1).to_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\dynamic_generation]\x.csv')
# pd.DataFrame(dy.y_predictions_2).to_csv(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\code_running_results\dynamic_generation]\y.csv')
# dy.email_reminder()

# we have a problme here, why the real value is about 0.33 but the first prediction is about 0.30 twice
# figure out why the first prediction is always 0.3 instead of 0.33eV
# figure out why the prediction for Et1 is always smaller than the real value, check the scalor.
# the scalor seems to be fine, maybe because of the particular validation we are having tend to be predicted smaller
# lets check it with larger validation set (100), if we still have Et1 prediction less than Et2, then we must have a systematic error.
# seems we do have a systematic error:
# 1. just focus on the first step.
# 2. check the scalour.
# first step works!
# lets try to do validation.
# check is there anything wrong with the second step? maybe because it does not have any column name for second step training?
# %%-

df1.email_reminder()
