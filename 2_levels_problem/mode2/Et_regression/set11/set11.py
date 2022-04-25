# %%-- Imports
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
from MLobject_tlevel import *
# one doping level:
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\set11_1e15.csv', 'bandgap1', 5)
# np.shape(df1.data)
# multiple doping level:
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\set11_diff_doping.csv', 'bandgap1', 5)
# n type doping:
 #df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\set11_1e15_ntype.csv', 'bandgap1', 5)
# %%-

# %%-- different data engineering
# multiplying lifetime by (dn+doping)
df1.pre_processor_dividX()
# %%-

# %%-- Single tasks.
# %%-- Perform regression for Et single task.
df1.singletask = 'Et_eV_1'
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
# %%-

# %%-- Data leakage.
# %%-- Regression for Et2 known Et1 and Et1+Et.
df1.singletask = 'Et_eV_2_known_Et_eV_2_plus'
r2scores = df1.regression_repeat()
# this makes the results better but has data leakage, R2 got about 0.999.
# %%-
# %%-

# %%-- Perform chain regression for energy levels.

# %%-- Just the chain.
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et2')
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et1+Et2->Et2')
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et1+Et2->logk_1->logk_1+logk_2->Et2')
# pd.DataFrame(np.array(chain_scores).reshape(35, 2)).to_csv(path_or_buf = r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\chainscore_two_steps.csv')
# %%-

# %%-- Chain and subtraction.
# the plan is to first predict Et1, then predict Et1+Et2, then predict Et2 by subtracting the prediction of sum by Et1 prediction.
r2 = df1.sum_minus_Et1_chain(regression_order=None, plotall=True)
# %%-

# %%-

# %%-- Perform chain regression for k
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'logk1+logk2->logk1->logk2')
# %%-
