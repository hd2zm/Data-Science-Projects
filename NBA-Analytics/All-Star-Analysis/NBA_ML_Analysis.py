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

print(NBA_MVA_df.head(15))

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

print(NBA_MVA_df.head(15))

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(NBA_MVA_df[['ALL-STAR', 'HEIGHT', 'STANDING REACH', 'WEIGHT', 'WINGSPAN']], hue="ALL-STAR")

plt.show()
plt.clf()


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
Random Forest Classifier on Height and Wingspan
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(NBA_MVA_df[['WINGSPAN', 'HEIGHT']], NBA_MVA_df['ALL-STAR'], test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
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
plt.title('Random Forest Classifier')
plt.xlabel('Wingspan')
plt.ylabel('Height')
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
plt.title('Random Forest Classifier')
plt.xlabel('Wingspan')
plt.ylabel('Height')
plt.legend()
plt.show()

#Predict Kawhi Leonard and Draymond Green Measurements
print(sc.transform([[87, 78],[85.25, 77.75]]))
print(classifier.predict(sc.transform([[87, 78],[85.25, 77.75]])))

#Predict Ideal Measurements : 6ft 8 inches  height with a 7ft 6 inches wingspan
print(sc.transform([[90,80]]))
print(classifier.predict(sc.transform([[90,80]])))


'''
PCA + Random Forest Classifier on Weight, Standing Reach, Height and Wingspan
'''


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = NBA_MVA_df.loc[:, ['HEIGHT', 'STANDING REACH', 'WEIGHT', 'WINGSPAN']].values
X = sc.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponent = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponent, columns = ['PCA1', 'PCA2'])
principalDf = pd.concat([principalDf, NBA_MVA_df.loc[:,['ALL-STAR']]], axis=1)
print(principalDf.head(30))

sns.scatterplot(x="PCA1", y="PCA2", hue="ALL-STAR", data=principalDf)

plt.show()
plt.clf()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(principalDf[['PCA1', 'PCA2']], principalDf['ALL-STAR'], test_size = 0.25, random_state = 0)


# Feature Scaling


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''

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
plt.title('Random Forest Classifier')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
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
plt.title('Random Forest Classifier')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
