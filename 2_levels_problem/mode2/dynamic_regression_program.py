# %%--- introduction
'''
Since the previous version of dynamic data generation method using object does not work, and the reason is unkown.
This file will write everything again from scrach and in a program instead of an object, so it will be easier to debug
'''
# %%-

# %%---import libraries:
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, f1_score, accuracy_score, mean_squared_error
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
from MLobject_tlevel import *
import sys
# sys.path.append(r'C:\Users\budac\Documents\GitHub\Yoann_code\DPML')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\DPML')
# from Si import *

# use this line for dell laptop:
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new')
# use this line for workstation:
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new\DPML')
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new')
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
from email.message import EmailMessage
import os
# %%-


# %%-- some simple predefined function: try not to make too many layers of function, so it is easier to debug.
def transparency_calculator(datasize):
    '''
    This function will calcualte a suitable data transparency given the datasize for a scatter plot.

    input: datasize: an integer.
    '''
    if datasize>800:
        alpha = 800/datasize*0.5
    else:
        alpha = 0.5
    return alpha
# %%-


# %%--- Step 1: train and test the Et_eV_1, logS_n1, logS_p1.
'''
Plan:
1. load the data.
2. take the lifetime of dataset 0
3. take the log10 of hte lifetime from dataset1
4. define X=log10(lifetime).
5. Define the scaler and go through the scaler and save the scaler.
6. define X and y
7. train test split.
8. train and save the model (use random forest and only one repetition)
9. output the prediciton, the model, the scaler and plot real vs predicted on the validation set.
'''

# load the data:
# the dataset0 (the real validation set)
dataset0 = pd.read_csv(r"G:\study\thesis_data_storage\set11\set11_8k.csv")
# the dtaset1 (the dataset that varies everytyhing to train Et1, Sn1, Sp1)
dataset1 = pd.read_csv(r"G:\study\thesis_data_storage\set11\set11_80000.csv")

# select the lifeitme columes, the criteria is whether the colume title start with a number:
select_X_list = []
for string in dataset1.columns.tolist():
    if string[0].isdigit():
        select_X_list.append(string)
X_dataset1 = dataset1[select_X_list]

# take the log10 of the lifetime from dataset1
X_log_dataset1 = np.log10(X_dataset1)

# define the scaler:
scaler = MinMaxScaler()
# train the scaler and let the processed lifetime go through the scaler.
X_log_scaled_dataset1 = scaler.fit_transform(X_log_dataset1)

# create a list to collect the models:
model_Et1_logSn1_logSp1 = []

# iterate for each y task.
for taskname in ['Et_eV_1', 'logSn_1', 'logSp_1']:

    # define the y for the ML model.
    y = dataset1[taskname]

    # train test split:
    X_train, X_test, y_train, y_test = train_test_split(X_log_scaled_dataset1, y, test_size=0.1)

    # define, train and save the model.
    model = RandomForestRegressor()
    print('training ' + str(model) + ' for ' + str(taskname))
    model.fit(X_train, y_train)
    model_Et1_logSn1_logSp1.append(model)

    # plot the real vs prediction for validation set.
    y_predict = model.predict(X_test)
    r2=r2_score(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    plt.figure()
    # print(np.shape(r2_frame))
    # print(np.shape(y_prediction_frame))
    # print(np.shape(y_test_frame))
    # print(np.shape(y_prediction_frame))
    # calculate the transparency:
    alpha=transparency_calculator(len(y_predict))
    print('transparency of scattering plot is ' + str(alpha))
    plt.scatter(y_test, y_predict, label=('$R^2$' + '=' + str(round(r2, 3))) + ('  Mean Absolue error' + '=' + str(round(mae, 3))), alpha=alpha)
    plt.xlabel('real value')
    plt.ylabel('predicted value')
    plt.title('real vs predicted at trial ' + ' for task ' + str(taskname))
    plt.legend()
    plt.show()

model_step1 = model_Et1_logSn1_logSp1 # this is a list a model.
scaler_step1 = scaler # this is just one scaler.
# %%-


# %%-- Step2: take the model 1 to predict Et1, Sn1, Sp1 from the validation point.
'''
plan:
1. load the validationset.
2. select the lifetime data of the validationset.
3. take the log of the vlidationset.
4. take the X through the scaler we had.
5. use the trained model to predict the processed lifetime.
6. return the prediction.
7. plot the real vs predicted. (next step is to apply step 2 on a single validation datapoint) (leave iteration through each validation point till the end)
'''

# load the validationset
# the dataset0 (the real validation set)
dataset0 = pd.read_csv(r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_1.csv")

# select the lifetime data of hte validationste by choosing hte colume names start with a number.
# select the lifeitme columes, the criteria is whether the colume title start with a number:
select_X_list = []
for string in dataset0.columns.tolist():
    if string[0].isdigit():
        select_X_list.append(string)
X_dataset0 = dataset0[select_X_list]

# take the log of validationset X.
X_log_dataset0 = np.log10(X_dataset0)

y_list = []
# iterate through each task:
for k in range(len(model_step1)):
    # select the correct machine learning model.
    model = model_step1[k]
    # select the correct scaler.
    scaler = scaler_step1
    # process the data through the scaler.
    X_log_scaled_dataset0 = scaler.transform(X_log_dataset0)
    # ask ML to predict y
    y_pred_set0 = model.predict(X_log_scaled_dataset0)
    # collect the prediction into hte list: it should be [Et1_predicted, logSn1_predicted, logSp1_predicted]
    y_list.append(y_pred_set0)

print(y_list)
# %%-


# %%-- Email reminder
def email_reminder():

    subject='ML finish training'
    body='ML finished' + ' through the file ' + str(os.getcwd())
    to='z5183876@ad.unsw.edu.au'

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
