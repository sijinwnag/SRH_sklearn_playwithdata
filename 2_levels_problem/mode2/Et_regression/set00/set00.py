# %%-- Imports
import sys
# import the function file from another folder:
# use this line if on hp laptop:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
from MLobject_tlevel import *

df1 = MyMLdata_2level(r"G:\study\thesis_data_storage\set00\set00_800k.csv", 'bandgap1',2)
# %%-

# %%-- different data engineering before training ML model.
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
df1.singletask = 'logSn_2'
r2scores = df1.regression_repeat()
df1.singletask = 'logSp_2'
r2scores = df1.regression_repeat()
df1.email_reminder()
# %%-

# %%-- Two level behaviour tester
df1.singletask = 'Et_eV_2'
r2scores = df1.regression_repeat()
df1.singletask = 'logSn_2'
r2scores = df1.regression_repeat()
df1.singletask = 'logSp_2'
r2scores = df1.regression_repeat()
df1.email_reminder()
# %%-

# %%-- First level behaviour tester
df1.singletask = 'Et_eV_1'
r2scores = df1.regression_repeat()
df1.singletask = 'logSn_1'
r2scores = df1.regression_repeat()
df1.singletask = 'logSp_1'
r2scores = df1.regression_repeat()
df1.email_reminder()
# %%-

# %%-

# %%-- Data leakage.
df1.singletask = 'Et_eV_2_known_Et_eV_2_plus'
r2scores = df1.regression_repeat()
# this makes the results better but has data leakage, R2 got about 0.999.

df1.singletask = 'Et_eV_2_known_param1'
r2scores = df1.regression_repeat()
df1.email_reminder()
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
# this eqution works for lifetime data that vary both T and doping.
# df1.C1n_C2n_C1d_C2d_calculator(return_C=False, export=True, sanity_check=True, playmusic=True)
# %%-

# %%-- Data visualization
# %%-- General historam:
# histogram for C:
df1.C_visiaulization(variable='C1n/C2n')
df1.C_visiaulization()
# %%-
# %%-- Histogram for different T.
# plot for one temperauree:
# df1.C_visiaulization(task_name='histogram at T', T=150)

# plot for demoninator term.
for T in range(150, 401, 50):
    df1.C_visiaulization(task_name='histogram at T', T=T)
# plot for numeator term:
for T in range(150, 401, 50):
    df1.C_visiaulization(task_name='histogram at T', variable='C1n/C2n', T=T)
# why it seems like T does not change anything?
# %%-
# %%-- Histogram for different doping.
df1.C_visiaulization(task_name='histogram at doping', doping=1e14)
# # plot for demoninator term.
# for doping in [3e14, 7e14, 1e15, 3e15, 7e15, 1e16]:
#     df1.C_visiaulization(task_name='histogram at doping', doping=doping)
# # plot for numeator term:
# for doping in [3e14, 7e14, 1e15, 3e15, 7e15, 1e16]:
#     df1.C_visiaulization(task_name='histogram at doping', doping=doping, variable='C1n/C2n')
# # why it seems like doping does not change anything either.
# %%-
# %%-- Visialize individual parameters.
df1.C_visiaulization(variable='C1n')
df1.C_visiaulization(variable='C2n')
df1.C_visiaulization(variable='C1d')
df1.C_visiaulization(variable='C2d')
df1.C_visiaulization(task_name='C histogram compare')
# %%-
# %%-- T vs C:
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with T')
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with T')
# %%-
# %%-- Doping vs C:
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with doping')
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with doping')
# %%-
# %%-- dn vs C:
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with dn')
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with dn')
# %%-
# %%-- E_diff vs C:
df1.C_visiaulization(variable='C1n/C2n', task_name='plot with Et1-Et2')
df1.C_visiaulization(variable='C1d/C2d', task_name='plot with Et1-Et2')
# %%-
# %%-
