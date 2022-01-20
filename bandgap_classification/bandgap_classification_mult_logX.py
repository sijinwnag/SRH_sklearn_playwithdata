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
import sys

# import the function file from another folder:
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata')
from function_for_trainings import classification_repeat, classification_training
##################################################################################
# firstly, load the data:
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# identify extract the useful columns
# delete all the defect information except bandgap
delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'logk', 'Et_eV']
dfk = df.drop(delete_col, axis=1)

# train test split:
X = dfk.drop(['bandgap'], axis=1)
y = dfk['bandgap']

# model training and evaluation
f1scores = classification_repeat(X, y, 1)

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
