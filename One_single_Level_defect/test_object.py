import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
from MLobject import *

df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\2022-02-01-12-07-56_Main_datasetID_0.csv', 'Et_eV', 5)
# df1.regression_repeat(n_repeat=1) # check if k regression works after converting into class.

# score_standard = df1.perform_alltasks_ML()
# export score 1
# score_standard[0].to_csv('k_originalsetting.csv')
# score_standard[1][0].to_csv('Et_overall_originalsetting.csv')
# score_standard[1][1].to_csv('Et_plus_originalsetting.csv')
# score_standard[1][2].to_csv('Et_minus_originalsetting.csv')
# score_standard[2].to_csv('classification_originalsetting.csv')

# df1.pre_processor_insert_dtal()
# score_dtau = df1.perform_alltasks_ML()
# export score dtau:
# score_dtau[0].to_csv('k_dtau.csv')
# score_dtau[1][0].to_csv('Et_overall_dtau.csv')
# score_dtau[1][1].to_csv('Et_plus_dtau.csv')
# score_dtau[1][2].to_csv('Et_minus_dtau.csv')
# score_dtau[2].to_csv('classification_dtau.csv')
