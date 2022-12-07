# %%-- import library.
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
import sys
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from playsound import playsound
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import RegressorChain
from semiconductor.recombination import SRH
import scipy.constants as sc
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
from sklearn.inspection import permutation_importance
import sympy as sym
import matplotlib.font_manager as font_manager
# %%-


# %%--
# Load the data from Yan's code export
Et1_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\Et1_alone.csv'
Et1_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et1_dominate\two_level.csv'
Et2_one_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\Et2_one_level_defect.csv'
Et2_two_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_dominate\two_level_defect.csv'
Et2_0_1_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0_1.csv'
Et2_0_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et2_0.csv'
Et12_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_01.csv'
Et12_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et12_both_n_01.csv'
Et1_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_0_1.csv'
Et1_n_01_path = r'C:\Users\sijin wang\Desktop\research\thesiswork\thesis3\Et1_Et2_cover\Et2_not_sensitive\Et1_n_0_1.csv'

Et1_one_lifetime = pd.read_csv(Et1_one_path)
Et1_two_lifetime = pd.read_csv(Et1_two_path)
Et2_one_lifetime = pd.read_csv(Et2_one_path)
Et2_two_lifetime = pd.read_csv(Et2_two_path)
Et2_0_1_lifetime = pd.read_csv(Et2_0_1_path)
Et2_0_lifetime = pd.read_csv(Et2_0_path)
Et12_01 = pd.read_csv(Et12_01_path)
Et12_n_01 = pd.read_csv(Et12_n_01_path)
Et1_01 = pd.read_csv(Et1_01_path)
Et1_n_01 = pd.read_csv(Et1_n_01_path)
# %%-


# %%-- Plot the Et1 dominate case.
plt.figure(facecolor='white', figsize=(5, 5))
plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t1}$')
plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect')
print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
plt.legend(fontsize=13)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
plt.ylabel('Lifetime (s)', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('$E_{t1}=0.5 eV$; $E_{t2}=0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
plt.savefig('Et1_dominate.png', bbox_inches='tight')
plt.show()
# print(np.max(Et1_one_lifetime.iloc[:, 1]-Et1_two_lifetime.iloc[:, 1]))
plt.figure(facecolor='white')
plt.plot(Et1_one_lifetime.iloc[:, 0], Et1_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t1}$ only')
plt.plot(Et1_two_lifetime.iloc[:, 0], Et1_two_lifetime.iloc[:, 1], label='Two level defect')
plt.legend(fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
plt.ylabel('Lifetime (s)', fontsize=22)
# plt.title('$E_{t1}=0.5 eV$; $E_{t2}=0.5 eV$' )
plt.xlim([10**(16.9), 1e17])
plt.ylim([1e-5, 10**-4.6])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
plt.savefig('Et1_dominate_zoomin.png', bbox_inches='tight')
plt.show()
# %%-


# %%--Plot Et2 dominate case.
plt.figure(facecolor='white', figsize=(5, 5))
plt.plot(Et2_one_lifetime.iloc[:, 0], Et2_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t2}$')
plt.plot(Et2_two_lifetime.iloc[:, 0], Et2_two_lifetime.iloc[:, 1], label='Two level defect')
plt.legend(fontsize=13)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Excess carrer concentration ($\rm cm^{-3}$)', fontsize=22)
plt.ylabel('Lifetime(s)', fontsize=22)
# plt.title('$E_{t1}=-0.5 eV$; $E_{t2}=-0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('Et2_dominate.png', bbox_inches='tight')
plt.show()

plt.figure(facecolor='white')
plt.plot(Et2_one_lifetime.iloc[:, 0], Et2_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t2}$ only')
plt.plot(Et2_two_lifetime.iloc[:, 0], Et2_two_lifetime.iloc[:, 1], label='Two level defect')
plt.legend(fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Excess carrer concentration ($cm^{-3}$)', fontsize=22)
plt.ylabel('Lifetime(s)', fontsize=22)
# plt.title('$E_{t1}=-0.5 eV$; $E_{t2}=-0.5 eV$' )
plt.xlim([10**(16.9), 1e17])
plt.ylim([1e-5, 10**-4.6])
plt.savefig('Et2_dominate_zoomin.png', bbox_inches='tight')
plt.show()
# %%-


# %%-- Plot Et2 not sensitive.
plt.figure(facecolor='white', figsize=(5, 5))
plt.plot(Et2_0_1_lifetime.iloc[:, 0], Et2_0_1_lifetime.iloc[:, 1], label=r'E$_{t2}$ = 0.1 eV')
plt.plot(Et2_0_lifetime.iloc[:, 0], Et2_0_lifetime.iloc[:, 1], label=r'E$_{t2}$ = 0 eV')
plt.plot(Et12_01.iloc[:, 0], Et12_01.iloc[:, 1])
plt.plot(Et1_01.iloc[:, 0], Et1_01.iloc[:, 1])
plt.plot(Et1_n_01.iloc[:, 0], Et1_n_01.iloc[:, 1])
print(np.max(Et2_0_1_lifetime.iloc[:, 1]-Et2_0_lifetime.iloc[:, 1]))
font = font_manager.FontProperties(family='Cambria', style='normal', size=20)
# plt.legend(framealpha=0.1, prop=font)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'Excess carrer concentration (cm$^{-3}$)', fontsize=22, font='Cambria')
plt.ylabel('Lifetime(s)', fontsize=22, font='Cambria')
# plt.title('$E_{t1}=-0.5 eV$; $E_{t2}=-0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
plt.xticks(fontsize=20, font='Cambria')
plt.yticks(fontsize=20, font='Cambria')
plt.savefig('Et2_dominate.png', bbox_inches='tight')
plt.show()

# plt.figure(facecolor='white')
# plt.plot(Et2_one_lifetime.iloc[:, 0], Et2_one_lifetime.iloc[:, 1], label='One level defect lifetime of $E_{t2}$ only')
# plt.plot(Et2_two_lifetime.iloc[:, 0], Et2_two_lifetime.iloc[:, 1], label='Two level defect')
# font = font_manager.FontProperties(family='Cambria', style='normal', size=20)
# plt.legend(framealpha=0.1, prop=font)
# # plt.legend(fontsize=15)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Excess carrer concentration ($cm^{-3}$)', fontsize=22, font='Cambria')
# plt.ylabel('Lifetime(s)', fontsize=22, font='Cambria')
# # plt.title('$E_{t1}=-0.5 eV$; $E_{t2}=-0.5 eV$' )
# plt.xlim([10**(16.9), 1e17])
# plt.ylim([1e-5, 10**-4.6])
# plt.savefig('Et2_dominate_zoomin.png', bbox_inches='tight')
# plt.show()
# %%-