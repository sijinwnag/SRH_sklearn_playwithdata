# %%-- To do:
"""

"""
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
df1 = MyMLdata_2level(r'C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\yoann_code_new\Savedir_example\outputs\set00.csv', 'bandgap_1',5)
# %%-

# %%-- preprocessors.

# %%-- equal number of set 11 set 10 and set 00:
# generate set 11 set 10 and set 00 saperately with same size.
# then integrate and shuffle them together.
path1 = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set00.csv"
path2 = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set10.csv"
path3 = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11.csv"
df1.dataset_integrator(path1, path2, path3)
# %%-

# %%-- multiply by (dn+p0+n0)
df1.pre_processor_dividX()
# %%-

# %%-

# %%-- Classification method: Et1
# classify Et1
df1.singletask = 'bandgap_1'
f1scores = df1.classification_repeat(display_confusion_matrix=True)
# %%-

# %%-- classify Et2 given Et1>0
path2 = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set10.csv"
path3 = r"C:\Users\sijin wang\Desktop\Thesis\thesiswork\simulation_data\set11.csv"
df1.dataset_integrator2(path2, path3)
df1.singletask = 'bandgap_2'
f1scores = df1.classification_repeat()
df1.email_reminder()
# %%-

# %%-- implement the multi-class classification task for both energy levels.
df1.singletask = 'multi_class_Et'
f1scores = df1.classification_repeat(display_confusion_matrix=True)
'''
Figure out the columes for the confusion matrix mean for multi-class, the third row behaves bad, figure out what it is:
the first row correspond to the true 0
the secont row correspond to the true 1
the third row correspond to the true 2 (which is set 11)
The confusion matrix told us:
1. it does a good job classifying set 10 when it is set 10.
3. if the defect belongs to set 00, it is likely to be misclassfied as 10.
2. if the defect belongs to set 11, it is likely to be misclassified as 10
'''
df1.email_reminder()
# %%-

# %%-- Identify whether the defect is 10
df1.singletask = 'whether 10'
f1scores = df1.classification_repeat(display_confusion_matrix=True)
# %%-

# %%-- Identify whether the defect is 11
df1.classfilter(filterclass=1)
# sanity check: expect to see a set of 0 and 2 but not 1
# df1.data['bandgap_1']+df1.data['bandgap_2']
df1.singletask = 'whether 11'
f1scores = df1.classification_repeat(display_confusion_matrix=True)
df1.email_reminder()
# %%-
