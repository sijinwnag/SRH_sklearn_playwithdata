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
df = pd.read_csv(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\lifetime_dataset_example.csv')

# identify extract the useful columns
# delete all the defect information except bandgap
delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'logk', 'Et_eV']
dfk = df.drop(delete_col, axis=1)

# train test split:
X = dfk.drop(['bandgap'], axis=1)
y = dfk['bandgap']

# model training and evaluation
f1scores = classification_repeat(X, y, 5)
# create barchart to compare f1 scores.
avf1scores = np.average(f1scores, axis=0)

# now, make logX into X and redo the process, compare the f1 scores:
X = np.log(X)
f1_scoreslog = classification_repeat(X, y, 5)
avf1scoreslog = np.average(f1_scoreslog, axis=0)

models_names = ['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network']
df_plot = pd.DataFrame({'using original X': avf1scores, 'using logX': avf1scoreslog}, index=models_names)
ax = df_plot.plot.barh()
ax.legend(bbox_to_anchor=(1.4, 0.55))
plt.title('the R2 scores for training using original X vs using logX')
