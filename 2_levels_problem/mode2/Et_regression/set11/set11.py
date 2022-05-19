# %%-- Imports
import sys

# import the function file from another folder:
# use this line if on hp laptop:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
# sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
from MLobject_tlevel import *

# %%-- use this secion if using dell laptop

# one doping level: 1e15, varying T, p, 8000 datapoints
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_1e15.csv', 'bandgap1', 10)
# np.shape(df1.data)

# multiple doping level: varying T, doping, p, 8000 datapoints
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_diff_doping.csv', 'bandgap1', 5)

# n type doping: vary T, 1e15, n, 8000 datapoints
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\set11_1e15_ntype.csv', 'bandgap1', 5)

# all known data: add temperature, doping and excess carrier cocnentration as extra columns.
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\new_data.csv', 'bandgap1', 5)

# one doping one temperature, p type:
# 1e14 doping and 400K, p type, 8000 data points, all files below works:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\one_doping_one_temp.csv', 'bandgap1', 50)
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\400K_1e14_8000_p.csv', 'bandgap1', 50)
# 1e15 dopiong 300K, p type: both files below works
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\1e15_300K.csv', 'bandgap1', 5)
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\300K_1e15_8000_p.csv', 'bandgap1', 5)
# 1e15 doping 400K, p type:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\400K_1e15_8000_p.csv', 'bandgap1', 5)
# 1e14 300K, p type 8000 data points:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\300K_1e14_8000_p.csv', 'bandgap1', 5)
# 1e15 200K, ptype, 8000 datapoints:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\200K_1e15_8000_p.csv', 'bandgap1', 5)
# 1e15 100K, p type, 8000 datapoints:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\100K_1e15_8000_p.csv', 'bandgap1', 5)
# 1e15 150K ptype 8000 datapints:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11\single_T_single_doping\p\150K_1e15_8000_p.csv', 'bandgap1', 5)
# %%-

# %%--use this section if using hp laptop:
# one doping level:
# df1 = MyMLdata_2level(r'C:\ML_databank\set11_1e15.csv', 'bandgap1', 10)
# df1 = MyMLdata_2level(r'"C:\ML_databank\set11_1e15.csv"', 'bandgap1', 10)
# %%-

# %%--use this section if using workstation
# df1 = MyMLdata_2level(r'', 'bandgap1', 10)
# %%-
# %%-

# %%-- different data engineering
# multiplying lifetime by (dn+p0+n0)
df1.pre_processor_dividX()
# %%-

# %%-- Single tasks.
# %%-- Perform regression for Et single task.
df1.singletask = 'Et_eV_1'
df1.regression_matrix = 'Mean Absolute Error'
r2scores = df1.regression_repeat()
df1.singletask = 'Et_eV_2'
r2scores = df1.regression_repeat() # R2 about 0.2 to 0.4
df1.singletask = 'Et_eV_1+Et_eV_2'
r2scores = df1.regression_repeat()
df1.singletask = 'Et_eV_1-Et_eV_2'
r2scores = df1.regression_repeat()
# %%-

# %%-- Perform regression for k single tasks.
df1.singletask = 'logk_1'
r2scores = df1.regression_repeat()
df1.singletask = 'logk_2'
r2scores = df1.regression_repeat()
df1.singletask = 'logk_1+logk_2'
r2scores = df1.regression_repeat()
df1.singletask = 'logk_1-logk_2'
r2scores = df1.regression_repeat()
# %%-

# %%-- Perform regression for sigma single tasks.
df1.singletask = 'logSn_1'
r2scores = df1.regression_repeat()
df1.singletask = 'logSp_1'
r2scores = df1.regression_repeat()
# %%-
# %%-

# %%-- Data leakage.
# %%-- Regression for Et2 known Et1 and Et1+Et.
df1.singletask = 'Et_eV_2_known_Et_eV_2_plus'
r2scores = df1.regression_repeat()
# this makes the results better but has data leakage, R2 got about 0.999.

df1.singletask = 'Et_eV_2_known_Et_eV_1'
r2scores = df1.regression_repeat()
# %%-
# %%-

# %%-- Perform chain regression for energy levels.

# %%-- Just the chain.
df1.regression_matrix = 'Mean Absolute Error'
df1.regression_matrix = 'R2'
chain_scores = df1.repeat_chain_regressor(repeat_num=3, regression_order=None, chain_name = 'Et1->Et2')
chain_scores = df1.repeat_chain_rDegressor(repeat_num=10, regression_order=None, chain_name = 'Et1->Et1+Et2->Et2')
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et1+Et2->logk_1->logk_1+logk_2->Et2')
# pd.DataFrame(np.array(chain_scores).reshape(35, 2)).to_csv(path_or_buf = r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\chainscore_two_steps.csv')
# %%-

# %%-- Chain and subtraction.
# the plan is to first predict Et1, then predict Et1+Et2, then predict Et2 by subtracting the prediction of sum by Et1 prediction.
# r2 = df1.sum_minus_Et1_chain(regression_order=None, plotall=True)
model_names, y_pred_matrix, y_test, r2list = df1.repeat_subtraction_method(repeat_num=5, regression_order=None, plotall=False, return_pred=True)
# %%-

# %%-

# %%-- Perform chain regression for k
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'logk1+logk2->logk1->logk2')
# %%-

# %%-- insert all known information as columns (failed)
df1.pre_processor_insert_all_known()
# %%-

# %%-- calculate C1 C2 C3 C4 as known for each defect.
# This equation only calcualte C while ignoring excess carrier concentration, and only works for one doping and one temperature.
# df1.C1_C2_C3_C4_calculator()
# C2n_frame, C2d_frame, C1n_frame, C1d_frame = df1.C1n_C2n_C1d_C2d_calculator(return_C=False, export=False)
# %%-

# %%-- Data visualization
# histogram for C:
df1.C_visiaulization(task_name='histogram at T')
# %%-
