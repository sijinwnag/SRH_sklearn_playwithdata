# %%-- To do:
'''
1. Figure out the columes for the confusion matrix mean for multi-class, the third row behaves bad, figure out what it is.
'''
# %%-

# %%-- Imports and define the object.
import sys
# import the function file from another folder:
# use this line if on hp laptop:
sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on dell laptop
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
# use this line if on workstation
sys.path.append(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\SRH_sklearn_playwithdata\2_levels_problem\mode2')
from MLobject_tlevel import *
# define the object
df1 = MyMLdata_2level(r'C:\Users\sijin wang\Documents\GitHub\yoann_code_new\Savedir_example\outputs\2022-06-06-14-54-01_advanced example - multi_level_L_datasetID_0.csv', 'bandgap_1',5)
# %%-

# %%-- implement the classification task for Et1.
df1.singletask = 'bandgap_1'
f1scores = df1.classification_repeat(display_confusion_matrix=True)
# %%-

# %%-- implement the classirfication task for Et2.
df1.singletask = 'bandgap_2'
f1scores = df1.classification_repeat()
df1.email_reminder()
# %%-

# %%-- implement the multi-class classification task for both energy levels.
df1.singletask = 'multi_class_Et'
f1scores = df1.classification_repeat(display_confusion_matrix=True)
df1.email_reminder()
# %%-
