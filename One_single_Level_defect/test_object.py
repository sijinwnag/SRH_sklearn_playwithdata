# %%--imports
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
from MLobject import *
df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\2022-02-01-12-07-56_Main_datasetID_0.csv', 'Et_eV', 1)
# df1.regression_repeat(n_repeat=1) # check if k regression works after converting into class.
# %%-

# %%--- original setting as Yoann code: single task.
# df1.singletask = 'k'
# df1.regression_repeat()
df1.singletask = 'bandgap'
df1.classification_repeat()
# %%-

# %%---add derivative as X.
# df1.pre_processor_insert_dtal()
# score_dtau = df1.perform_alltasks_ML()
# export score dtau:
# score_dtau[0].to_csv('k_dtau.csv')
# score_dtau[1][0].to_csv('Et_overall_dtau.csv')
# score_dtau[1][1].to_csv('Et_plus_dtau.csv')
# score_dtau[1][2].to_csv('Et_minus_dtau.csv')
# score_dtau[2].to_csv('classification_dtau.csv')
# %%-

# %%--- chain regressor.
# chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=[0, 1])
# chain_scores[0].to_csv('k_chain1_kfirst.csv')
# chain_scores[1].to_csv('Eplus_chain1_kfirst.csv')
# chain_scores[2].to_csv('Eminus_chain1_kfirst.csv')
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=[1, 0])
chain_scores[0].to_csv('k_chain1_Efirst.csv')
chain_scores[1].to_csv('Eplus_chain1_Efirst.csv')
chain_scores[2].to_csv('Eminus_chain1_Efirst.csv')
# %%-

# %%--- Data visualization
# investigate the correlation between the y values: we have k and Et
df1.mypairplot(['Et_eV', 'k'])
# from the plot we see that they are not very correlated this make sense because the Et_eV and k are from simulated data,
# and during simulation we generate them randomly
# %%-
