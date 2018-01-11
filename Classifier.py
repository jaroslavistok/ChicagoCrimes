from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from DataWrangler import DataWrangler


class Classifier:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.accuracy_train_score = 0
        self.accuracy_test_score = 0
        self.cross_validation_train_score = 0
        self.cross_validation_test_score = 0
        self.classififier_pipeline = None

    def train_and_predict(self, classifier):
        self.classififier_pipeline = Pipeline((
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ))

        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        self.classififier_pipeline.fit(X_train, y_train)
        self.accuracy_train_score = accuracy_score(y_train, self.classififier_pipeline.predict(X_train))
        self.accuracy_test_score = accuracy_score(y_test, self.classififier_pipeline.predict(X_test))
        self.cross_validation_train_score = np.mean(cross_val_score(self.classififier_pipeline, X_train, y_train, cv=10))
        self.cross_validation_test_score = np.mean(cross_val_score(self.classififier_pipeline, X_test, y_test, cv=10))

    def print_results(self):
        print("Train accuracy: " + str(self.accuracy_train_score))
        print("Test accuracy: " + str(self.accuracy_test_score))
        print("Cross validation train score: " + str(self.cross_validation_train_score))
        print("Cross validation test score: " + str(self.cross_validation_test_score))


