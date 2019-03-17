#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:50:18 2019

@author: hdeva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:47:05 2019

@author: hdeva
"""

import pandas as pd
import numpy as np

NBA_MVA_df = pd.read_csv("NBA_Measurements_Value_Added.csv").dropna()
NBA_MVA_df = NBA_MVA_df.drop(['Unnamed: 0'], axis=1)
NBA_MVA_df = NBA_MVA_df[NBA_MVA_df.WINGSPAN > 50]
NBA_MVA_df = NBA_MVA_df[NBA_MVA_df.WEIGHT != '-']
NBA_MVA_df['WEIGHT'] = pd.to_numeric(NBA_MVA_df['WEIGHT'] , downcast='float')
NBA_MVA_df = NBA_MVA_df[NBA_MVA_df.WEIGHT > 0]

NBA_MVA_df['STANDING REACH'] = pd.to_numeric(NBA_MVA_df['STANDING REACH'] , downcast='float')
NBA_MVA_df['HEIGHT'] = pd.to_numeric(NBA_MVA_df['HEIGHT'] , downcast='float')
NBA_MVA_df['VA'] = pd.to_numeric(NBA_MVA_df['VA'] , downcast='float')
NBA_MVA_df['WINGSPAN'] = pd.to_numeric(NBA_MVA_df['WINGSPAN'] , downcast='float')


NBA_MVA_df= NBA_MVA_df.reset_index()
NBA_MVA_df = NBA_MVA_df.drop(['index'], axis=1)

print(NBA_MVA_df)

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(NBA_MVA_df[['ALL-STAR', 'HEIGHT', 'STANDING REACH', 'WEIGHT', 'WINGSPAN', 'VA']], hue="ALL-STAR")

plt.show()
plt.clf()

#fig, ax = plt.subplots(figsize=(15,15)) 
#sns.heatmap(NBA_MVA_df.corr(),annot=True, linewidths=.5, ax=ax)

sns.scatterplot(x="WINGSPAN", y="VA", hue="ALL-STAR", data=NBA_MVA_df)

plt.show()
plt.clf()

sns.scatterplot(x="HEIGHT", y="VA", hue="ALL-STAR", data=NBA_MVA_df)

plt.show()
plt.clf()

sns.scatterplot(x="STANDING REACH", y="VA", hue="ALL-STAR", data=NBA_MVA_df)

plt.show()
plt.clf()

sns.scatterplot(x="WEIGHT", y="VA", hue="ALL-STAR", data=NBA_MVA_df)

plt.show()
plt.clf()

sns.scatterplot(x="WINGSPAN", y="HEIGHT", hue="ALL-STAR", data=NBA_MVA_df)

plt.show()
plt.clf()





'''
Naive Bayes analysis on Wingspan and VA - because people say wingspan accounts for success
'''


'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(NBA_MVA_df[['WINGSPAN', 'VA']], NBA_MVA_df['ALL-STAR'], test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Wingspan')
plt.ylabel('Value Added')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Wingspan')
plt.ylabel('Value Added')
plt.legend()
plt.show()


'''



'''
PCA on combinations that makes a player an all star
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = NBA_MVA_df.loc[:, ['HEIGHT', 'STANDING REACH', 'WEIGHT', 'WINGSPAN']].values
X = sc.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
principalComponent = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponent, columns = ['PCA'])
principalDf = pd.concat([principalDf, NBA_MVA_df.loc[:,['VA', 'ALL-STAR']]], axis=1)
print(principalDf.head(30))


'''
Naive Bayes analysis on PCA and VA 
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(principalDf[['PCA', 'VA']], principalDf['ALL-STAR'], test_size = 0.25, random_state = 0)

# Feature Scaling


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('PCA')
plt.ylabel('Value Added')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('PCA')
plt.ylabel('Value Added')
plt.legend()
plt.show()

