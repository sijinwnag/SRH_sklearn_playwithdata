# %%--- Imports.
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem')
from MLobject_tlevel import *
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\single_two_level_prob.csv', 'bandgap', 5)
# %%-

# %%--- perform bandgap classification.
df1.singletask='bandgap_1'
# try svc by itself
# X, y = df1.pre_processor()
f1scores, y_prediction_frame, y_test_frame = df1.classification_repeat(display_confusion_matrix=False, output_y_pred=True)
# playsound('spongbob.mp3')
f1scores.to_csv('bandgap1')

# do the same for bandgap_2
df1.singletask='bandgap_2'
# try svc by itself
# X, y = df1.pre_processor()
f1scores2, y_prediction_frame, y_test_frame = df1.classification_repeat(display_confusion_matrix=False, output_y_pred=True)
# playsound('spongbob.mp3')
# f1scores2.to_csv('bandgap1')
# does not work
# %%-

# %%--- Perform k regression.
df1.singletask = 'logk_1'
r2scores = df1.regression_repeat()
# not working
df1.singletask = 'logk_2'
r2scores = df1.regression_repeat()
# not working.
df1.singletask = 'logk_1+logk_2'
r2scores = df1.regression_repeat()
# r2scores.to_csv('logk1pluslogk2')
# %%-

# %%--- Perform Et regression.
df1.singletask = 'Et_eV_1'
r2scores = df1.regression_repeat()
# does not work.
df1.singletask = 'Et_eV_2'
r2scores = df1.regression_repeat()
# does not work
df1.singletask = 'Et_eV_1+Et_eV_2'
r2scores = df1.regression_repeat()
# does not work
# %%-
