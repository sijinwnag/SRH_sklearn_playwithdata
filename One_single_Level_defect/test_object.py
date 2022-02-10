import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
from MLobject import *

df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\lifetime_dataset_example.csv', 'Et_eV', 5)
# df1.regression_repeat(n_repeat=1) # check if k regression works after converting into class.
scores = df1.perform_alltasks_ML()
# scores[0].to_csv('k_regression.csv')
# scores[1].to_csv('Et_regression.csv')
# scores[2].to_csv('bandgap_classification.csv')
df1.singletask = 'Et_plus'
scores = df1.regression_repeat()
