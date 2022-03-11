# %%--- Imports.
import sys
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem')
from MLobject_tlevel import *
# %%-

# %%--- Load the data.
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\single_two_level_prob.csv', 'bandgap', 5)
# %%-

# %%--- perform bandgap classification.
df1.singletask='bandgap_1'
# try svc by itself
# X, y = df1.pre_processor()
f1scores, y_prediction_frame, y_test_frame = df1.classification_repeat(display_confusion_matrix=False, output_y_pred=True)
playsound('spongbob.mp3')
# %%-
