#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:18:42 2019

@author: hdeva
"""



federal_folder = 'federal_data'
mk_folder = 'mk_data'
federal_coc_pit_file = '2007-2018-PIT-Counts-by-CoC'
federal_state_pit_file = '2007-2018-PIT-Counts-by-State'
mk_pit_file = 'Cumulative_MK_PIT_Cleaned_Results_2007_to_2018'

import pandas as pd
from pandas import Series,DataFrame

federal_coc_pit_excel_file = pd.ExcelFile('data/' + federal_folder + '/' + federal_coc_pit_file + '.xlsx')
federal_state_pit_excel_file = pd.ExcelFile('data/' + federal_folder + '/' + federal_state_pit_file + '.xlsx')

mk_pit_excel_file = pd.ExcelFile('data/' + mk_folder + '/' + mk_pit_file + '.xlsx')

mk_pit_data = pd.read_excel(mk_pit_excel_file, 'Sheet2')
mk_pit_data = {str(key).replace(' ', ''): val  
     for key, val in mk_pit_data.items()} 

state_key = 7
coc_key = 68

ratio_th_dict = {}

ratio = lambda subset,total: subset/total 

for mk_key in mk_pit_data.keys():
    if 'Unnamed' not in str(mk_key):
    
        federal_state_pit_data = pd.read_excel(federal_state_pit_excel_file, str(mk_key))
        federal_coc_pit_data = pd.read_excel(federal_coc_pit_excel_file, str(mk_key))
	
        ratio_th_dict[mk_key] = {}
	
        ratio_th_dict[mk_key]['Federal State PIT'] = ratio(federal_state_pit_data['Sheltered TH Homeless, ' + str(mk_key)][state_key], federal_state_pit_data['Overall Homeless, ' + str(mk_key)][state_key])
        ratio_th_dict[mk_key]['Federal CoC PIT'] = ratio(federal_coc_pit_data['Sheltered TH Homeless, ' + str(mk_key)][coc_key], federal_coc_pit_data['Overall Homeless, ' + str(mk_key)][coc_key]) 
        #ratio_th_dict[mk_key]['MK PIT'] = ratio(int(float(mk_pit_data[str(mk_key)]['Shelter / Transitional Housing']) * int(mk_pit_data[str(mk_key)]['Surveys Completed'])), int(mk_pit_data[str(mk_key)]['Surveys Completed']))
        ratio_th_dict[mk_key]['MK PIT'] = float(mk_pit_data[str(mk_key)][47]) 



ratio_th_pit_df = pd.DataFrame(data = ratio_th_dict).T

print(ratio_th_pit_df.head())

import seaborn as sns

sns.violinplot(x="PIT Categories", y="Sheltered TH Homeless PIT Ratio", data=pd.melt(ratio_th_pit_df, var_name='PIT Categories', value_name='Sheltered TH Homeless PIT Ratio'))

sns.swarmplot(x="PIT Categories", y="Sheltered TH Homeless PIT Ratio", data=pd.melt(ratio_th_pit_df, var_name='PIT Categories', value_name='Sheltered TH Homeless PIT Ratio'), color='k', alpha=0.7)


import matplotlib.pyplot as plt
plt.show()

plt.clf()

sns.kdeplot(ratio_th_pit_df['MK PIT'])
sns.kdeplot(ratio_th_pit_df['Federal State PIT'])
sns.kdeplot(ratio_th_pit_df['Federal CoC PIT'])

plt.show()

# Conduct 2 sample t test of means
from scipy import stats

t, p = stats.ttest_ind(ratio_th_pit_df['MK PIT'].tolist(), ratio_th_pit_df['Federal State PIT'].tolist(), None, False)
print('MK and Federal State PIT: T-stat=%.3f, p-val=%.3f' % (t, p))

t_2, p_2 = stats.ttest_ind(ratio_th_pit_df['MK PIT'].tolist(), ratio_th_pit_df['Federal CoC PIT'].tolist(), None, False)
print('MK and Federal CoC PIT: T-stat=%.3f, p-val=%.3f' % (t_2, p_2))


'''
# Chi-squared test with similar proportions 
from scipy import stats

# Contingency table
stat,p,dof,expected = stats.chi2_contingency([ratio_th_pit_df['MK PIT'].tolist(), ratio_th_pit_df['Federal State PIT'].tolist()])

# Interpret test-statistic
prob = 0.95
critical = stats.chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret P-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
    

# Contingency table
stat,p,dof,expected = stats.chi2_contingency([ratio_th_pit_df['MK PIT'].tolist(), ratio_th_pit_df['Federal CoC PIT'].tolist()])

# Interpret test-statistic
prob = 0.95
critical = stats.chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret P-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
'''