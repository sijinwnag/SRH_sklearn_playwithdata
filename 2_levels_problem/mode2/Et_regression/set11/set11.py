# %%-- Imports
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
from MLobject_tlevel import *
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\set11_1e15.csv', 'bandgap1', 5)
# np.shape(df1.data)
# %%-

# %%-- Perform regression for Et single task.
df1.singletask = 'Et_eV_1'
r2scores = df1.regression_repeat() # R2 above 0.9
df1.singletask = 'Et_eV_2'
r2scores = df1.regression_repeat() # R2 about 0.2 to 0.4
# %%-

# %%-- Perform regression for k single tasks.
df1.singletask = 'logk_1'
r2scores = df1.regression_repeat()
df1.singletask = 'logk_2'
r2scores = df1.regression_repeat()
# %%-

# %%-- Regression for Et2 known Et1 and Et1+Et.
df1.singletask = 'Et_eV_2_known_Et_eV_2_plus'
r2scores = df1.regression_repeat()
# this makes the results better but has data leakage, R2 got about 0.999.
# %%-

# %%-- Perform chain regression: Et1->Et1+Et2->Et or Et1->Et2
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et2', plotall=True)
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et1+Et2->Et2', plotall=True)
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'Et1->Et1+Et2->logk_1->logk_1+logk_2->Et2')
# pd.DataFrame(np.array(chain_scores).reshape(35, 2)).to_csv(path_or_buf = r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\Et_regression\set11\chainscore_two_steps.csv')
# %%-

# %%-- Perform chain regression for k
chain_scores = df1.repeat_chain_regressor(repeat_num=5, regression_order=None, chain_name = 'logk1+logk2->logk1->logk2')
# %%-

# %%-- Perform regression for k without chain.
# when doping is 1e15.
# the R2 for logk1 is about 0.89 with linear regression.
# the R2 for logk2 is about 0.55 using gradient boosting.
# the R2 for logk1+logk2 is about 0.831 using NN
for task in ['logk_1', 'logk_2', 'logk_1+logk_2']:
    df1.singletask = task
    r2score = df1.regression_repeat()
# %%-

# %%-- Data visualization
# %%-- Et distribution
# visualize the Et distribution by plotting the histogram for Et1 and Et2.
Et1 = df1.data['Et_eV_1']
Et2 = df1.data['Et_eV_2']
plt.figure()
plt.hist(Et1, label='$E_{t1}$', bins=50)
plt.hist(Et2, label='$E_{t2}$', bins=50)
plt.legend()
plt.title('Histogram of energy levels')
plt.show()
# %%-
# %%-- k distribution
# visualie the k distribution using histogram.
k1 = df1.data['logk_1']
k2 = df1.data['logk_2']
plt.figure()
plt.hist(k1, label='$logk_{1}$', bins=50)
plt.hist(k2, label='$logk_{2}$', bins=50)
plt.legend()
plt.title('Histogram of capture cross section ratios')
plt.show()
# %%-
# %%-
