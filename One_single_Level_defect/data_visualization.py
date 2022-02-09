import sys
import seaborn as sn
from playsound import playsound
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
from MLobject import *

df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\lifetime_dataset_example.csv', 'Et_eV', 5)

# view the columns
df1.data.columns

# for each defect data, we have 3 variables: doping level, temperature and excess carrier concentration
# to do good data visualization we fix the parameter one by one.
# select 3 columns: one as the reference, one change T another change excess carrier concentration
plot_col = ['300K_5100000000000000.0cm-3_ 10000000000000.0cm-3', '300K_5100000000000000.0cm-3_ 1e+17cm-3', '400K_5100000000000000.0cm-3_ 10000000000000.0cm-3', 'Et_eV', 'k', 'bandgap']
dfplot = df1.data[plot_col]
dfplot = dfplot.rename(columns={'300K_5100000000000000.0cm-3_ 10000000000000.0cm-3': '300K(5.1e15cm-3)(1e13cm-3)', '300K_5100000000000000.0cm-3_ 1e+17cm-3': '300K(5.1e15cm-3)(1e17cm-3)', '400K_5100000000000000.0cm-3_ 10000000000000.0cm-3':'400K(5.1e15cm-3)(1e17cm-3)'})

figure = sn.pairplot(dfplot)
figure.savefig('save_as_a_png.png')
playsound('spongbob.mp3')
