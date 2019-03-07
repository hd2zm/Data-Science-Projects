#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:57:21 2019

@author: hdeva
"""


mk_folder = 'mk_data'
mk_detailed_spdat_file = 'MK Detailed SPDAT Report'

import pandas as pd
from pandas import Series,DataFrame
import math


mk_detailed_spdat_excel_file = pd.ExcelFile('data/' + mk_folder + '/' + mk_detailed_spdat_file + '.xlsx')

mk_spdat_data = pd.read_excel(mk_detailed_spdat_excel_file, 'SPDATs Overall')

mk_spdat_df = pd.DataFrame(data = mk_spdat_data)

mk_spdat_df.drop(['Unnamed: 0'], axis = 1, inplace=True)

mk_spdat = {}

for index, row in mk_spdat_df.iterrows():
    if not math.isnan(row['Participant Site Identifier']):
        
        key = int(row['Participant Site Identifier'])

        if math.isnan(row['Score Change']):
            mk_spdat[key] = {}
            mk_spdat[key]['Beginning Score'] = row['Total Score']
            mk_spdat[key]['New'] = 1
        else:
            mk_spdat[key]['Ending Score'] = row['Total Score']
            mk_spdat[key]['Score Change'] = row['Score Change']
            mk_spdat[key]['New'] = 0
        
mk_spdat_df = pd.DataFrame(data = mk_spdat).T
mk_spdat_df.drop(mk_spdat_df[mk_spdat_df['New'] == 1].index, inplace=True)
mk_spdat_df.drop(['New'], axis = 1, inplace=True)
mk_spdat_df['Ending Score Predicted'] = mk_spdat_df['Beginning Score'] + mk_spdat_df['Score Change'] 
mk_spdat_df_scores = mk_spdat_df.drop(['Score Change'], axis = 1)

scatter_mk_spdat_df = mk_spdat_df_scores.melt('Beginning Score', var_name='endscores', value_name='Ending Score')

import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='Beginning Score', y='Ending Score', data=mk_spdat_df_scores)
plt.show()
plt.clf()
sns.scatterplot(x='Beginning Score', y='Ending Score Predicted', data=mk_spdat_df_scores)
plt.show()
plt.clf()

sns.kdeplot(mk_spdat_df_scores['Ending Score Predicted'])
sns.kdeplot(mk_spdat_df_scores['Ending Score'])

plt.show()
plt.clf()

# Using the elbow method to find the optimal number of clusters


X = mk_spdat_df_scores.drop(['Ending Score Predicted'], axis=1).values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'orange', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'purple', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Beginning Score')
plt.ylabel('Ending Score')
plt.legend()
plt.show()