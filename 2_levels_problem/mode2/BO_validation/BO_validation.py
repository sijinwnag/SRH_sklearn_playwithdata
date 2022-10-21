# %%--
'''
trial 1 for 800k: Et=0.3 eV, it is very far from the real value. Maybe the problem is from dataset.
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
sys.stdout = open(r"Bo_validation_Et1.txt", "w")
# load the BO example.
BO_data = pd.read_csv(r'G:\study\thesis_data_storage\unordered\yan_compare\BO\testset.csv')
# load the trianing data.
training_data = pd.read_csv(r'G:\study\thesis_data_storage\unordered\set10\p\800k\set10_800k_p.csv')

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
y_train = training_data['Et_eV_1']
y_test = BO_data['Et_eV_1']

# define the model.
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1)
# train the model.
model.fit(training_scaled, y_train)
# predict
print(model.predict(BO_scaled))
sys.stdout.close()
# %%- Et_eV_1

# %%--Et2
sys.stdout = open(r"Bo_validation_Et2.txt", "w")
# load the BO example.
BO_data = pd.read_csv(r'G:\study\thesis_data_storage\unordered\yan_compare\BO\testset.csv')
# load the trianing data.
training_data = pd.read_csv(r'G:\study\thesis_data_storage\unordered\set10\p\800k\set10_800k_p.csv')

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
    subject='data generation done'
    # email body
    body= 'data generation is done' + ' through the file ' + str(os.getcwd())
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
