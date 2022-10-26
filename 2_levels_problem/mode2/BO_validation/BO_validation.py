# %%--
'''
Et1 regression:

n-type:
Try with 8k data: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.197 (true value is 0.15).

p-type:
Try with 8k data: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.145 (true value is 0.15).
Try with 80k data: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.129 (true value is 0.15).
Try with 80k data trial 2: expect the Et1 prediction to be at least accurate: Et1 is predicted to be 0.138 (true value is 0.15). The ML seems to always underestimate it.
Try with 80k data trial 3, this time change the number of tree from 100 to 200. Et1 is predicted to be 0.143 (true value is 0.15). The ML seems to always underestimate it, but better this time.
Try with 80k data trial 4, number of tree being 300. Et1=0.14.
Try with 80k data trial 5,number of tree is 200.
'''

'''
One two level defect classification:
expect to be mode: single two-value.

trial 1: single two-value.
trial 2: single two-value.
trial 3: single two-value.
trial 4: single two-value.
trial 5: single two-value.
'''
# %%-

# %%--
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
import math
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
# %%-

# %%--Et1

# load the BO example.
BO_data = pd.read_csv(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\BO_validation\BO_ptype\2022-10-25-11-14-51_advanced example - multi_level_L_datasetID_0.csv')
# load the trianing data.
training_data = pd.read_csv(r'G:\study\thesis_data_storage\unordered\set10\p\80k\2022_10_25\2022-10-25-13-57-56_advanced example - multi_level_L_datasetID_0.csv')

# extract the lifetime.
BO_lifetime = BO_data.iloc[:,17:-2]
training_lifetime = training_data.iloc[:, 17:-2]
# BO_lifetime.head()
# training_lifetime.head()


# take log10
BO_lifetime_log = BO_lifetime.applymap(math.log10)
training_lifetime_log = training_lifetime.applymap(math.log10)


# go through scaler.
scaler = MinMaxScaler()
training_scaled = scaler.fit_transform(training_lifetime_log)
BO_scaled = scaler.transform(BO_lifetime_log)


# define the target variable.
y_train = training_data['Et_eV_1']
y_test = BO_data['Et_eV_1']
# y_train
# y_test

# define the model.
model = RandomForestRegressor(n_estimators=200, verbose=20)
# train the model.
model.fit(training_scaled, y_train)
# predict
Et1_prediction = model.predict(BO_scaled)
print(Et1_prediction)
# sys.stdout = open(r"Bo_validation_Et1.txt", "w")
# sys.stdout.close()
# %%- Et_eV_1

# %%--Et2
sys.stdout = open(r"Bo_validation_Et2.txt", "w")
# load the BO example.
BO_data = pd.read_csv(r'G:\study\thesis_data_storage\unordered\yan_compare\BO\testset.csv')
# load the trianing data.
training_data = pd.read_csv(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new\Savedir_example\outputs\2022-10-23-21-19-53_advanced example - multi_level_L_datasetID_0.csv')

# extract the lifetime.
BO_lifetime = BO_data.iloc[:,17:-2]
training_lifetime = training_data.iloc[:, 17:-2]

# take log10
BO_lifetime_log = np.log10(BO_lifetime)
training_lifetime_log = np.log10(training_lifetime)

# go through scaler.
scaler = MinMaxScaler()
training_scaled = scaler.fit_transform(training_lifetime_log)
BO_scaled = scaler.transform(BO_lifetime_log)

# define the target variable.
y_train = training_data['Et_eV_2']
y_test = BO_data['Et_eV_2']

# define the model.
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1)
# train the model.
model.fit(training_scaled, y_train)
# predict
print(model.predict(BO_scaled))
sys.stdout.close()
# %%- Et_eV_2

# %%--defect classification: one or two level.
# load the BO example.
BO_data = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\BO_validation\BO_ptype\2022-10-25-11-14-51_advanced example - multi_level_L_datasetID_0.csv')
# load the trianing data.
training_data = pd.read_csv(r'C:\Users\sijin wang\Desktop\research\thesiswork\ML_results\simulation_data\Etnonordered\one_vs_two_level_classification\80k\2022_09_29\2022-09-29-09-45-08_advanced example - multi_level_L_datasetID_0.csv')

# extract the lifetime.
BO_lifetime = BO_data.iloc[:,17:-2]
training_lifetime = training_data.iloc[:, 17:-2]
# BO_lifetime.head()
# training_lifetime.head()


# take log10
BO_lifetime_log = BO_lifetime.applymap(math.log10)
training_lifetime_log = training_lifetime.applymap(math.log10)


# go through scaler.
scaler = MinMaxScaler()
training_scaled = scaler.fit_transform(training_lifetime_log)
BO_scaled = scaler.transform(BO_lifetime_log)


# define the target variable.
y_train = training_data['Label']
y_test = BO_data['Label']
y_train
y_test

# define the model.
model = MLPClassifier((100, 100),alpha=0.001, activation = 'relu',verbose=10,learning_rate='adaptive')
# train the model.
model.fit(training_scaled, y_train)
# predict
print(model.predict(BO_scaled))
sys.stdout = open(r"Bo_validation_Et1.txt", "w")
sys.stdout.close()
# %%-

# %%-- Email reminder:
# becuase google disable allowing less secure app, the code below does not work anymore.
# def email_reminder():
#
#     # who to send to. and the content of the email.
#     subject='data generation done'
#     body= 'data generation is done' + ' through the file ' + str(os.getcwd())
#     to='z5183876@ad.unsw.edu.au'
#
#     # email address to be sent from: (you can use this address to send email from)
#     user = "sijinwang944@gmail.com"
#     password = 'vjvlqydqtxlpddgz'
#
#     msg = EmailMessage()
#     msg.set_content(body)
#     msg['subject'] = subject
#     msg['to'] = to
#     msg['from'] = user
#
#     server = smtplib.SMTP("smtp.gmail.com", 587)
#     server.starttls()
#     server.login(user, password)
#     server.send_message(msg)
#
#     server.quit()

def email_reminder():

    # who to send to. and the content of the email.
    # email title
    subject='BO test is done'
    # email body
    body= 'BO test is done' + ' through the file ' + str(os.getcwd()) + 'Et1 prediction is ' + str(Et1_prediction)
    # which email address to sent to:
    to='z5183876@ad.unsw.edu.au'

    # email address to be sent from: (you can use this address to send email from)
    user = "sijinwang@yahoo.com"
    password = 'gdfkzhzhaokjivek'

    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to
    msg['from'] = user

    server = smtplib.SMTP("smtp.mail.yahoo.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)

    server.quit()

email_reminder()
# %%-
