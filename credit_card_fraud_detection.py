import pandas as pd

data = pd.read_csv("creditcard.csv")

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

amount = data['Amount'].values
data['Amount'] = scalar.fit_transform(amount.reshape(-1,1))

data.drop(['Time'], axis = 1, inplace = True)
data.drop_duplicates(inplace=True)

X = data.drop(['Class'], axis = 1).values
y = data['Class'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, classification_report

parameters_knn = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15], 'metric': ["manhattan", "chebyshev", "hamming"], 'weights': ["uniform", "distance"]}
knn = KNeighborsClassifier()
knn_clf = GridSearchCV(knn, parameters_knn, cv=5)
knn_clf.fit(X_train, y_train)
print(knn_clf.best_params_)
knn_best = knn_clf.best_estimator_
knn_prediction = knn_best.predict(X_test)
print(classification_report(y_test, knn_prediction))