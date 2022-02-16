# %%-- Imports
import sys
import seaborn as sn
import pandas as pd
# import the function file from another folder:
# sys.path.append(r'C:\Users\budac\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
sys.path.append(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect')
from MLobject import *
from decimal import Decimal
# %%-

# %%-- Load the dataframe
df1 = MyMLdata(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\lifetime_dataset_example.csv', 'Et_eV', 5)
# %%-

# %%-- Selecting plotting columes
# for each defect data, we have 3 variables: doping level, temperature and excess carrier concentration
# to do good data visualization we fix the parameter one by one.
# select 3 columns: one as the reference, one change T another change excess carrier concentration
plot_col = ['300K_5100000000000000.0cm-3_ 10000000000000.0cm-3', '300K_5100000000000000.0cm-3_ 1e+17cm-3', '400K_5100000000000000.0cm-3_ 10000000000000.0cm-3', 'Et_eV', 'k', 'bandgap']
dfplot = df1.data[plot_col]
dfplot = dfplot.rename(columns={'300K_5100000000000000.0cm-3_ 10000000000000.0cm-3': '300K(5.1e15cm-3)(1e13cm-3)', '300K_5100000000000000.0cm-3_ 1e+17cm-3': '300K(5.1e15cm-3)(1e17cm-3)', '400K_5100000000000000.0cm-3_ 10000000000000.0cm-3':'400K(5.1e15cm-3)(1e17cm-3)'})
# %%-

# %%-- plot the overall picture:
figure = sn.pairplot(dfplot)
# %%-

# %%-- Selecting plotting columns but choose logk instead of k
plot_col = ['300K_5100000000000000.0cm-3_ 10000000000000.0cm-3', '300K_5100000000000000.0cm-3_ 1e+17cm-3', '400K_5100000000000000.0cm-3_ 10000000000000.0cm-3', 'Et_eV', 'logk', 'bandgap']
dfplot = df1.data[plot_col]
dfplot = dfplot.rename(columns={'300K_5100000000000000.0cm-3_ 10000000000000.0cm-3': '300K(5.1e15cm-3)(1e13cm-3)', '300K_5100000000000000.0cm-3_ 1e+17cm-3': '300K(5.1e15cm-3)(1e17cm-3)', '400K_5100000000000000.0cm-3_ 10000000000000.0cm-3':'400K(5.1e15cm-3)(1e17cm-3)'})
# %%-

# %%-- plot the overall picture with logk:
figure = sn.pairplot(dfplot)
# %%-

# %%-- Observation comment
# the initial oberservation is that when we have a low excess carrrier concentration or high T, Et tend to be either very high or very low at larger lifetime.
# to avoid being over generalized about the topic, Let us further plot more graphs to verify if the observation is true in general.
# We first plot: Et for different temperature
# Then plot: Et for different excess carrier level.
# to make sure the assumption was indeed true, then go to the SRH equation to see if that make sense.
# then think about how can you make this trend more obvious and better for the ML to handle it.
# %%-

# %%-- Plot Et for differet temperature
# to check if the temperature is different or not, we generate a new set of data that has the same exces carrer denstity but 5 different temperatures and plot hte data.
dfT = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\2022-02-10-10-20-27_Main_datasetID_0.csv')
plot_col = ['200K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '250K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '300K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '350K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '400K_1000000000000000.0cm-3_ 10000000000000.0cm-3', 'Et_eV']
dfplotT = dfT[plot_col]
# to convert the numbers into scientific notation to be displayed during plotting, prepare an emptly list to collect the text after conversion
plot_sci_col = []
for text in plot_col[:-1]:
    # we need to write the numbers in scientific notation
    # first we split hte text with _
    temp, doping_density, excess_carrier_density = text.split('_')
    # for the converting ones, get rid of the unit
    doping_density = doping_density.split('c')[0]
    excess_carrier_density = excess_carrier_density.split('c')[0]
    # convert the doping and excess carrier density into scientific notation
    doping_sce = "{0:.2E}".format(Decimal(doping_density))
    excess_sce = "{0:.2E}".format(Decimal(doping_density))
    # now put all the text back together and put into the new text list.
    plot_sci_col.append(str(temp) + str(doping_sce) + '$cm^{-3}$' + str(excess_sce) +'$cm^{-3}$')
# now we got a list of name to display columns for X, add the Et
plot_sci_col.append('Et_eV')
dfplotT.columns = plot_sci_col
sn.set(font_scale=0.8)
figure = sn.pairplot(dfplotT)
# %%-

# %%--- Do the plotting using object funcions.
df1.mypairplot(plot_col = ['200K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '250K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '300K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '350K_1000000000000000.0cm-3_ 10000000000000.0cm-3', '400K_1000000000000000.0cm-3_ 10000000000000.0cm-3', 'Et_eV'])
# %%-
