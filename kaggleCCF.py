'''
Alex Lux

The challenge is to recognize fraudulent credit card transactions so that the customers of credit card companies are not charged for items that they did not purchase.

Main challenges involved in credit card fraud detection are:

1. Enormous Data is processed every day and the model build must be fast enough to respond to the scam in time.
2. Imbalanced Data i.e most of the transactions (99.8%) are not fraudulent which makes it really hard for detecting the fraudulent ones
3. Data availability as the data is mostly private.
4. Misclassified Data can be another major issue, as not every fraudulent transaction is caught and reported.
5. Adaptive techniques used against the model by the scammers.

How to tackle these challenges?

1. The model used must be simple and fast enough to detect the anomaly and classify it as a fraudulent transaction as quickly as possible.
2. Imbalance can be dealt with by properly using some methods which we will talk about in the next paragraph
3. For protecting the privacy of the user the dimensionality of the data can be reduced.
4. A more trustworthy source must be taken which double-check the data, at least for training the model.
5. We can make the model simple and interpretable so that when the scammer adapts to it with just some tweaks we can have a new model up and running to deploy.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
LABELS = ["Normal", "Fraud"]

df = pd.read_csv("creditcard.csv")

print("===============================================================================================")
print("======================================== DATA FRAME ===========================================")
print("===============================================================================================")
print(df.head())
print("===============================================================================================")
print("===================================== DATA DESCRIPTION ========================================")
print("===============================================================================================")
print(df.describe())
print()
print("===================================== NULL VALUES ========================================")
print(df.isnull().values.any())

count_classes = pd.value_counts(df['Class'], sort = True)
print(print("===================================== CLASS COUNT (99.8% Normal, 0.2% fraud) ========================================"))
print(count_classes)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

X = df.drop(['Class'], axis = 1)

y = df['Class']

X_data = X.values
y_data = y.values

X_data_sample = X.sample(frac=0.1, random_state=123)
y_data_sample = y.sample(frac=0.1 , random_state=123)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data_sample, y_data_sample, test_size=0.2, random_state=123)

# print("X train:\n", X_train)
# print("y train:\n", y_train)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

'''
parameters_knn = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15], 'metric': ["manhattan", "chebyshev", "hamming"], 'weights': ["uniform", "distance"]}
knn = KNeighborsClassifier()
knn_clf = GridSearchCV(estimator=knn, param_grid=parameters_knn, cv=5, n_jobs=-1, verbose=2)
knn_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR KNN CLASSIFIER: {knn_clf.best_params_}")
knn_best = knn_clf.best_estimator_
knn_predictions = knn_best.predict(X_valid)
print(classification_report(y_valid, knn_predictions))


parameters_dt = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(estimator=dt, param_grid=parameters_dt, cv=5, n_jobs=-1, verbose=2)
dt_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR DECISION TREE CLASSIFIER: {dt_clf.best_params_}")
dt_best = dt_clf.best_estimator_
dt_predictions = dt_best.predict(X_valid)
print(classification_report(y_valid, dt_predictions))

nb = GaussianNB()
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB_clf = GridSearchCV(estimator=nb, param_grid=params_NB, cv=5, n_jobs=-1, verbose=2) 
gs_NB_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR GAUSSIAN NAIVE BAYES CLASSIFIER: {gs_NB_clf.best_params_}")
nb_best = gs_NB_clf.best_estimator_
nb_predictions = nb_best.predict(X_valid)
print(classification_report(y_valid, nb_predictions))
'''

lr = LinearRegression()
parameters_lr = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
lr_clf = GridSearchCV(estimator=lr, param_grid=parameters_lr, cv=5, n_jobs=-1, verbose=2)
lr_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR LINEAR REGRESSION CLASSIFIER: {lr_clf.best_params_}")
lr_best = lr_clf.best_estimator_
lr_predictions = lr_best.predict(X_valid)
print(classification_report(y_valid, lr_predictions))

logistic_r = LogisticRegression()
# TODO
# ==> parameters_logistic_r = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
logistic_r_clf = GridSearchCV(estimator=lr, param_grid=parameters_logistic_r, cv=5, n_jobs=-1, verbose=2)
logistic_r_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR LINEAR REGRESSION CLASSIFIER: {logistic_r_clf.best_params_}")
logistic_r_best = logistic_r_clf.best_estimator_
logistic_r_predictions = logistic_r_best.predict(X_valid)
print(classification_report(y_valid, logistic_r_predictions))