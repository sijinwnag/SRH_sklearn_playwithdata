import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
from MLobject import *

df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\2022-02-01-12-07-56_Main_datasetID_0.csv', 'Et_eV', 5)
# df1.regression_repeat(n_repeat=1) # check if k regression works after converting into class.

# %%--- original setting as Yoann code.
# score_standard = df1.perform_alltasks_ML()
# export score 1
# score_standard[0].to_csv('k_originalsetting.csv')
# score_standard[1][0].to_csv('Et_overall_originalsetting.csv')
# score_standard[1][1].to_csv('Et_plus_originalsetting.csv')
# score_standard[1][2].to_csv('Et_minus_originalsetting.csv')
# score_standard[2].to_csv('classification_originalsetting.csv')
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
singletask = 'Et_plus'
# for now we make single taks same as task, in the future, we make task capable of doing multiple task.
# define the columns to be deleted for ML purposes
delete_col = ['Name', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp']
# drop these columns
dfk = (pd.DataFrame(df1.data)).drop(delete_col, axis=1)
# if we are doing Et regression, we need to do them for above and below bandgap saperately
if singletask == 'Et_plus':
    dfk = dfk[dfk['Et_eV']>0]
# define X and y based on the task we are doing.
dfk = pd.DataFrame(dfk)
X = np.log(dfk.drop(['logk', 'Et_eV', 'bandgap'], axis=1))
# scale the data:
for col in X.columns:
    # print(X[col])
    X[col] = MinMaxScaler().fit_transform(X[col].values.reshape(-1, 1))
y = dfk[['logk', 'Et_eV']]
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.1)
df1.chain_regression_once(X_train_scaled, X_test_scaled, y_train, y_test, regression_order=[0, 1])
# %%-

# %%--- Data visualization
# investigate the correlation between the y values: we have k and Et
df1.mypairplot(['Et_eV', 'k'])
# from the plot we see that they are not very correlated this make sense because the Et_eV and k are from simulated data,
# and during simulation we generate them randomly
# %%-
