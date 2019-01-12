#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:19:19 2019

@author: rohitgupta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("IRIS.csv")

dataset.isnull().sum()

X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

import seaborn as sns
sns.lmplot('sepal_width', 'petal_width', dataset, hue='species', fit_reg=False)
fig = plt.gcf()
plt.show()

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((150,1)).astype(int), values=X, axis=1)

X_opt = X[:, [1,3,4]]
classifier_OLS = sm.OLS(endog=y, exog=X_opt).fit()
classifier_OLS.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, scoring='accuracy', cv=10)
accuracies.mean()

from sklearn.model_selection import GridSearchCV
parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}
    ]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
