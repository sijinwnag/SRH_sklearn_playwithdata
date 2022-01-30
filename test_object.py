import sys
# import the function file from another folder:
sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata')
# sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata')
from MLobject import *

df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\lifetime_dataset_example.csv', 'Et_eV', 2)
# df1.regression_repeat(n_repeat=1) # check if k regression works after converting into class.
df1.perfor_singletask_ML(plot_graphs=True)

list(range(7))
