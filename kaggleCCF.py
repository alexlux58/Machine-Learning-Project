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

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import sys, os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


from sklearn.neighbors import KNeighborsClassifier
parameters_knn = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15], 'metric': ["manhattan", "chebyshev", "hamming"], 'weights': ["uniform", "distance"]}
knn = KNeighborsClassifier()
knn_clf = GridSearchCV(estimator=knn, param_grid=parameters_knn, cv=5, n_jobs=-1, verbose=1)
knn_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR KNN CLASSIFIER: {knn_clf.best_params_}")
knn_best = knn_clf.best_estimator_
knn_predictions = knn_best.predict(X_valid)
print(classification_report(y_valid, knn_predictions))

import sklearn.metrics as metrics
fpr1, tpr1, threshold = metrics.roc_curve(y_valid, knn_predictions)
roc_auc1 = metrics.auc(fpr1, tpr1)

knn_cm = confusion_matrix(y_valid, knn_predictions)
print(knn_cm)
plot_confusion_matrix(knn_clf, X_train_new, y_train_new)
plt.show()


from sklearn.tree import DecisionTreeClassifier
parameters_dt = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(estimator=dt, param_grid=parameters_dt, cv=5, n_jobs=-1, verbose=1)
dt_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR DECISION TREE CLASSIFIER: {dt_clf.best_params_}")
dt_best = dt_clf.best_estimator_
dt_predictions = dt_best.predict(X_valid)
print(classification_report(y_valid, dt_predictions))

fpr2, tpr2, threshold = metrics.roc_curve(y_valid, dt_predictions)
roc_auc2 = metrics.auc(fpr2, tpr2)

dt_cm = confusion_matrix(y_valid, dt_predictions)
print(dt_cm)
plot_confusion_matrix(dt_clf, X_train_new, y_train_new)
plt.show()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB_clf = GridSearchCV(estimator=nb, param_grid=params_NB, cv=5, n_jobs=-1, verbose=1) 
gs_NB_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR GAUSSIAN NAIVE BAYES CLASSIFIER: {gs_NB_clf.best_params_}")
nb_best = gs_NB_clf.best_estimator_
nb_predictions = nb_best.predict(X_valid)
print(classification_report(y_valid, nb_predictions))

fpr3, tpr3, threshold = metrics.roc_curve(y_valid, nb_predictions)
roc_auc3 = metrics.auc(fpr3, tpr3)

gnb_cm = confusion_matrix(y_valid, nb_predictions)
print(gnb_cm)
plot_confusion_matrix(gs_NB_clf, X_train_new, y_train_new)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
parameters_lr = {'fit_intercept':[True,False],'copy_X':[True, False]}
lr_clf = GridSearchCV(estimator=lr, param_grid=parameters_lr, cv=5, n_jobs=-1, verbose=1)
lr_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR LINEAR REGRESSION CLASSIFIER: {lr_clf.best_params_}")
lr_best = lr_clf.best_estimator_
lr_predictions = lr_best.predict(X_valid)
print(classification_report(y_valid, lr_predictions.round()))

fpr4, tpr4, threshold = metrics.roc_curve(y_valid, lr_predictions)
roc_auc4 = metrics.auc(fpr4, tpr4)

# lr_cm = confusion_matrix(y_valid, lr_predictions)
# print(lr_cm)
# plot_confusion_matrix(lr_clf, X_train_new, y_train_new)
# plt.show()

from sklearn.linear_model import LogisticRegression
logistic_r = LogisticRegression()
parameters_logistic_r = {'penalty':['none', 'l2', 'l1', 'elasticnet'], 'C':[0.01, 0.1, 0.5, 1], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
logistic_r_clf = GridSearchCV(estimator=logistic_r, param_grid=parameters_logistic_r, cv=5, n_jobs=-1, verbose=1)
logistic_r_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR LOGISTIC REGRESSION CLASSIFIER: {logistic_r_clf.best_params_}")
logistic_r_best = logistic_r_clf.best_estimator_
logistic_r_predictions = logistic_r_best.predict(X_valid)
print(classification_report(y_valid, logistic_r_predictions))

fpr5, tpr5, threshold = metrics.roc_curve(y_valid, logistic_r_predictions)
roc_auc5 = metrics.auc(fpr5, tpr5)

lgr_cm = confusion_matrix(y_valid, logistic_r_predictions)
print(lgr_cm)
plot_confusion_matrix(logistic_r_clf, X_train_new, y_train_new)
plt.show()

from sklearn.svm import LinearSVC
svc = LinearSVC()
parameters_svc = {'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'loss': ['hinge', 'squared_hinge'], 'C': [1 , 10 , 0.1]}
svc_clf = GridSearchCV(estimator=svc, param_grid=parameters_svc, cv=3, n_jobs=-1, verbose=1)
svc_clf.fit(X_train_new, y_train_new)
print(f"BEST PARAMETERS FOR LinearSVC CLASSIFIER: {svc_clf.best_params_}")
svc_best = svc_clf.best_estimator_
svc_predictions = svc_best.predict(X_valid)
print(classification_report(y_valid, svc_predictions))

fpr6, tpr6, threshold = metrics.roc_curve(y_valid, svc_predictions)
roc_auc6 = metrics.auc(fpr6, tpr6)

svc_cm = confusion_matrix(y_valid, svc_predictions)
print(svc_cm)
plot_confusion_matrix(svc_clf, X_train_new, y_train_new)
plt.show()

from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

# Number of features (first layer inputs)
n_inputs = 30

nn_model = Sequential()
# define first hidden layer and visible layer
nn_model.add(Dense(50, input_dim=n_inputs, activation='relu', kernel_initializer='he_uniform'))
# define output layer
nn_model.add(Dense(1, activation='sigmoid'))
# define loss and optimizer
nn_model.compile(loss='binary_crossentropy', optimizer='adam')

import datetime

log_dir = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
filepath = 'nn_model.hdf5'

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=3, save_best_only=True, mode='max')
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

nn_model.fit(X_train_new, y_train_new, epochs=10, callbacks=[checkpoint, tensorboard_callbacks])
eval = nn_model.evaluate(X_valid, y_valid)
print(f"EVALUATION: {eval}")

nn_predictions = nn_model.predict(X_valid)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_valid,nn_predictions))

nn_predictions_flat = nn_predictions.flatten()
y_pred = np.where(nn_predictions_flat > 0.5, 1, 0)

print(accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2Score = r2_score(y_valid, y_pred)
maeScore = mean_absolute_error(y_valid, y_pred)
mseScore = mean_squared_error(y_valid, y_pred)
print(f"R2: {r2Score}, MAE: {maeScore}, MSE: {mseScore}")

fpr7, tpr7, threshold = metrics.roc_curve(y_valid, y_pred)
roc_auc7 = metrics.auc(fpr7, tpr7)

nn_cm = confusion_matrix(y_valid, y_pred)
print(nn_cm)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'AUC for KNN = %0.2f' % roc_auc1)
plt.plot(fpr2, tpr2, 'r', label = 'AUC for Decision Tree = %0.2f' % roc_auc2)
plt.plot(fpr3, tpr3, 'g', label = 'AUC for Gaussian Naive Bayes = %0.2f' % roc_auc3)
plt.plot(fpr4, tpr4, 'c', label = 'AUC for Linear Regression = %0.2f' % roc_auc4)
plt.plot(fpr5, tpr5, 'y', label = 'AUC for Logistic Regression = %0.2f' % roc_auc5)
plt.plot(fpr6, tpr6, 'k', label = 'AUC for Linear SVC = %0.2f' % roc_auc6)
plt.plot(fpr7, tpr7, 'm', label = 'AUC for Neaural Network = %0.2f' % roc_auc7)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()