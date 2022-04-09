# %%-- Imports
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem')
from MLobject_tlevel import *
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\set11_1e15.csv', 'bandgap1', 1)
# np.shape(df1.data)
# %%-

# %%-- Perform regression for Et
df1.singletask = 'Et_eV_1'
r2scores = df1.regression_repeat() # R2 above 0.9
df1.singletask = 'Et_eV_2'
r2scores = df1.regression_repeat() # R2 about 0.2 to 0.4
# %%-

# %%-- Regression for Et2 known Et1 and Et1+Et.
df1.singletask = 'Et_eV_2_known_Et_eV_2_plus'
r2scores = df1.regression_repeat() # this makes the results better but has data leakage, test if adding the esmiated bandgap 1 will help: R2 got about 0.999.
# %%-

# %%-- Perform chain regression: Et1->Et1+Et2->Et2.
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None)
pd.DataFrame(np.array(chain_scores).reshape(35, 3)).to_csv('chainscores2')
