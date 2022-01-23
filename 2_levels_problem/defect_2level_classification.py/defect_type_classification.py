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
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\2level_defects.csv')

# identify extract the useful columns
# delete all the defect information except the defect class.
delete_col = ['Name', 'Et_eV_1', 'Sn_cm2_1', 'Sp_cm2_1', 'k_1', 'logSn_1', 'logSp_1', 'logk_1', 'bandgap_1', 'Et_eV_2', 'Sn_cm2_2', 'Sp_cm2_2', 'k_2', 'logSn_2', 'logSp_2', 'logk_2', 'bandgap_2']
dfk = df.drop(delete_col, axis=1)

# encode the column: Mode.
dfk['Mode'] = pd.Categorical(dfk['Mode'])
dfk = pd.get_dummies(dfk)
# dfk.columns.values.tolist()
dfk = dfk.drop(['Mode_Two one-level'], axis=1)
# train test split:
X = dfk.drop(['Mode_Single two-level'], axis=1)
y = dfk['Mode_Single two-level']

# now, make logX into X and redo the process, compare the f1 scores:
f1_scoreslog = classification_repeat(X, y, 5)
avf1scoreslog = np.average(f1_scoreslog, axis=0)
avf1scoreslog
models_names = ['KNN', 'SVC', 'Decision tree', 'Random Forest',  'Gradient Boosting', 'Adaptive boosting', 'Naive Bayes', 'Neural Network']
df_plot = pd.DataFrame({'using original X': avf1scores, 'using logX': avf1scoreslog}, index=models_names)
ax = df_plot.plot.barh()
ax.legend(bbox_to_anchor=(1.4, 0.55))
plt.title('the R2 scores for training using original X vs using logX')