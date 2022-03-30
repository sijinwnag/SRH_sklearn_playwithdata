# %%-- Imports.
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem')
from MLobject_tlevel import *
# df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\single_two_level_1e15doping.csv', 'bandgap1', 2)
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2\single_two_level_prob.csv', 'bandgap1', 2)
# %%-

# %%-- perform bandgap classification.
df1.singletask='bandgap_1'
# try svc by itself
# X, y = df1.pre_processor()
f1scores, y_prediction_frame, y_test_frame = df1.classification_repeat()
# playsound('spongbob.mp3')
# f1scores.to_csv('bandgap1')
# does not work: the micro f1 score is high because most of bandgap 1 is 1, so the score of guessing everything is 1 is high, if you calculate macro average scroe, the score is less than 0.5, which is low.

# do the same for bandgap_2
df1.singletask='bandgap_2'
# try svc by itself
# X, y = df1.pre_processor()
f1scores2, y_prediction_frame, y_test_frame = df1.classification_repeat()
# playsound('spongbob.mp3')
# f1scores2.to_csv('bandgap1.csv')
# does not work
# %%-

# %%-- Perform k regression.
df1.singletask = 'logk_1'
r2scores = df1.regression_repeat()
# r2scores.to_csv('k1_original.csv')
# not working
df1.singletask = 'logk_2'
r2scores = df1.regression_repeat()
# not working.
df1.singletask = 'logk_1+logk_2'
r2scores = df1.regression_repeat()
# r2scores.to_csv('logk1pluslogk2')
# %%-

# %%-- Perform Et regression.
df1.singletask = 'Et_eV_1'
r2scores = df1.regression_repeat()
# does not work.
df1.singletask = 'Et_eV_2'
r2scores = df1.regression_repeat()
# does not work
df1.singletask = 'Et_eV_1+Et_eV_2'
r2scores = df1.regression_repeat()
# does not work
# try asking for E1 given bandgap 1
df1.singletask = 'Et_eV_1_known_bandgap1'
r2scores = df1.regression_repeat() # this makes the results better but has data leakage, test if adding the esmiated bandgap 1 will help

# do regression for the difference of energy level: does not work, but better than doing each energy individually, but still very bad.
df1.singletask = 'Et_eV_1-Et_eV_2'
r2scores = df1.regression_repeat()
# %%-

# %%-- Performe 2 step regression: bandgap1->Et1: failed because bandgap1 does not work
# test the splitting function.
set1, set2 = df1.dataset_splitter() # now df1 data is the whole data.
# use the first set to predict the bandgap, return the model.
df1.data = set1
df1.singletask = 'bandgap_1'
f1_frame, y_prediction_frame, y_test_frame, bestmodel, scaler = df1.classification_repeat(display_confusion_matrix=False, return_model=True) # now df1 data is set 1
# use the best model to predicct bandgap_1 on set 2
df1.data = set2
y_pred = df1.apply_given_model(set2, 'bandgap_1', bestmodel, scaler)
# check the prediction accuracy.
# now add this extra column in to hte X data.
df1.data['predicted_bandgap_1'] = y_pred
f1_score(df1.data['bandgap_1'], y_pred, average='macro')
sum(y_pred)
sum(df1.data['bandgap_1'])
# now use the new dataset to predict Et1.
df1.singletask = 'Et_eV_1_known_predicted_bandgap_1'
r2scores = df1.regression_repeat()
# it turns out the bandgap 1 classification does not work either
# %%-

# %%-- Try splitting the dataset then try different tasks:
set11, set10, set01, set00 = df1.bandgap_split()
# %%--- Perform k regression for set 11
df1.data = set11
df1.singletask = 'logk_1'
r2scores_11_k1 = df1.regression_repeat()
# r2scores_11_k1.to_csv('k1_set11.csv')
# not work either same behaviour as original setting.
for dataset in [set00]:
    df1.data = dataset
    for task in ['logk_1', 'logk_2', 'logk_1+logk_2', 'Et_eV_1', 'Et_eV_2']:
        df1.singletask = task
        r2score = df1.regression_repeat()
# %%-


# %%--- Data visualization
df1.data = set11
df1.mypairplot(['Et_eV_1', '200K_1.01e+16cm-3_ 259502421139974.25cm-3'])
df1.data = set10
df1.mypairplot(['Et_eV_1', '200K_1.01e+16cm-3_ 259502421139974.25cm-3'])
# %%-
# %%-
