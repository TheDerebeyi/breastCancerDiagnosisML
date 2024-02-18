from abc import ABC, abstractmethod

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt

class Model(ABC):
    def __init__(self, X_train, y_train):
        self.param_grid = {}
        self.model = None
        self.X_train = X_train
        self.y_train = y_train

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def find_best_params(self):
        grid_search = GridSearchCV(self.model, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        self.model.set_params(**grid_search.best_params_)
        return grid_search.best_params_


class KNN(Model):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        self.param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        self.model = KNeighborsClassifier()

    def train(self):
        best_params = self.find_best_params()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class SVM(Model):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        self.param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        self.model = SVC()

    def train(self):
        best_params = self.find_best_params()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class NaiveBayes(Model):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        self.param_grid = {'alpha': [0.1, 0.5, 1.0]}
        self.model = MultinomialNB()

    def train(self):
        best_params = self.find_best_params()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class ModelEvaluation:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Reds', annot_kws={"size": 16})

        ax.set_xlabel('y_pred')
        ax.set_ylabel('y_true')
        ax.legend()

        return ax
