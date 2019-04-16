#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:20:40 2019

@author: hdeva
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd

rpm_next_url = 'http://www.espn.com/nba/statistics/rpm'
per_next_url = 'http://insider.espn.com/nba/hollinger/statistics'

# Set up empty data list
rpm_data = []
per_data = []

i = 1

max_rpm_page = 13
max_stat_page = 8

while i <= max_rpm_page:   
    #Set as Beautiful Soup Object
    rpm_soup = BeautifulSoup(requests.get(rpm_next_url).content)

    # Go to the section of interest
    rpm_summary = rpm_soup.find("div",{'class':'span-4', 'id':'my-players-table'})

    # Find the tables in the HTML
    rpm_tables = rpm_summary.find_all('table')
    
    # Set rows as first indexed object in tables with a row
    rows = rpm_tables[0].findAll('tr')

    # now grab every HTML cell in every row
    for tr in rows:
        cols = tr.findAll('td')
        # Check to see if text is in the row
        rpm_data.append([])
        for td in cols:
            text = td.find(text=True) 
            rpm_data[-1].append(text)
    
    i = i+1    
            
    try:
        rpm_next_url = 'http://www.espn.com/nba/statistics/rpm/_/page/' + str(i)
        
    except IndexError:
        break
    
i = 1

while i <= max_stat_page:   
    #Set as Beautiful Soup Object
    per_soup = BeautifulSoup(requests.get(per_next_url).content)

    # Go to the section of interest
    per_summary = per_soup.find("div",{'class':'col-main', 'id':'my-players-table'})

    # Find the tables in the HTML
    per_tables = per_summary.find_all('table')
    
    # Set rows as first indexed object in tables with a row
    rows = per_tables[0].findAll('tr')

    # now grab every HTML cell in every row
    for tr in rows:
        cols = tr.findAll('td')
        # Check to see if text is in the row
        per_data.append([])
        for td in cols:
            text = td.find(text=True) 
            per_data[-1].append(text)
    
    i = i+1    
            
    try:
        per_next_url = 'http://insider.espn.com/nba/hollinger/statistics/_/page/' + str(i)
    except IndexError:
        break
    
def removeRank(stat_list):
    return list(map(lambda stat_record: stat_record.pop(0), stat_list))

removeRank(rpm_data)
per_data.pop(0)
removeRank(per_data)

rpm_df = pd.DataFrame(rpm_data[1:], columns=rpm_data[0])
per_df = pd.DataFrame(per_data[1:], columns=per_data[0])
rpm_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)

metrics_df = pd.merge(rpm_df, per_df, how='left', on=['PLAYER', 'GP', 'MPG'])
metrics_df = metrics_df[metrics_df.PLAYER != 'NAME']
print(metrics_df.head(45))
print(list(metrics_df.columns))
metrics_df['GP'] = pd.to_numeric(metrics_df['GP'], downcast='integer')
metrics_df['MPG'] = pd.to_numeric(metrics_df['MPG'], downcast='float')
metrics_df['ORPM'] = pd.to_numeric(metrics_df['ORPM'], downcast='float')
metrics_df['DRPM'] = pd.to_numeric(metrics_df['DRPM'], downcast='float')
metrics_df['RPM'] = pd.to_numeric(metrics_df['RPM'], downcast='float')
metrics_df['WINS'] = pd.to_numeric(metrics_df['WINS'], downcast='float')
metrics_df['TS%'] = pd.to_numeric(metrics_df['TS%'], downcast='float')
metrics_df['AST'] = pd.to_numeric(metrics_df['AST'], downcast='float')
metrics_df['TO'] = pd.to_numeric(metrics_df['TO'], downcast='float')
metrics_df['USG'] = pd.to_numeric(metrics_df['USG'], downcast='float')
metrics_df['ORR'] = pd.to_numeric(metrics_df['ORR'], downcast='float')
metrics_df['DRR'] = pd.to_numeric(metrics_df['DRR'], downcast='float')
metrics_df['REBR'] = pd.to_numeric(metrics_df['REBR'], downcast='float')
metrics_df['PER'] = pd.to_numeric(metrics_df['PER'], downcast='float')
metrics_df['VA'] = pd.to_numeric(metrics_df['VA'], downcast='float')
metrics_df['EWA'] = pd.to_numeric(metrics_df['EWA'], downcast='float')
#print(metrics_df.dtypes)

#import seaborn as sns
import matplotlib.pyplot as plt

#Data Analysis
#fig, ax = plt.subplots(figsize=(15,15)) 
#sns.heatmap(metrics_df.corr(),annot=True, linewidths=.5, ax=ax)


mpg_rpm_correlation = metrics_df['MPG'].corr(metrics_df['RPM'])
print(mpg_rpm_correlation)
print(metrics_df['PER'].corr(metrics_df['RPM']))

#Reset indeces for sklearn
#metrics_df = metrics_df.reset_index()
metrics_df = metrics_df.fillna(0)

#Split for Multiple Linear Regression
X_high_correlation = metrics_df[['MPG','RPM','WINS','USG','PER']]
y = metrics_df[['VA']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_high_correlation, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)


#Getting Regression results using stat models
import statsmodels.api as sm
X_high_correlation = sm.add_constant(X_high_correlation)
model = sm.OLS(y, X_high_correlation).fit()
predictions = model.predict(X_high_correlation)
print(model.summary())

#Getting Regression results using stat models

X_high_correlation_two = metrics_df[['MPG','RPM','WINS','USG','PER']]
import statsmodels.api as sm
X_high_correlation = sm.add_constant(X_high_correlation_two)
model = sm.OLS(y, X_high_correlation_two).fit()
predictions = model.predict(X_high_correlation_two)
print(model.summary())
    
metrics_df['VA_Pred'] = metrics_df.apply(lambda metrics_df_record: regressor.predict([[metrics_df_record['MPG'],
                                          metrics_df_record['RPM'],
                                          metrics_df_record['WINS'],
                                          metrics_df_record['USG'],
                                          metrics_df_record['PER']]]), axis=1)
    
metrics_df = metrics_df.sort_values(by=['VA_Pred'], ascending=False)
#print(metrics_df.head(50))

metrics_df.to_csv('NBA_Adjusted_Value_Added.csv')


#sns.pairplot(metrics_df[['MPG','RPM','WINS','USG','PER', 'VA']])