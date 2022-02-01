import sys
# import the function file from another folder:
sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata')
from MLobject import *

df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\2022-01-31-17-16-48_Main_datasetID_0.csv', 'Et_eV', 1)
# df1.regression_repeat(n_repeat=1) # check if k regression works after converting into class.
scores = df1.perform_alltasks_ML()
scores
scores[0]
scores[1]
scores[2]
