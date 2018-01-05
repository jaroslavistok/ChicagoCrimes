
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import numpy
from DataWrangler import DataWrangler

class DecisionTree:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.decision_tree = None

        self.train_error = None
        self.test_error = None
        self.train_log_loss = None
        self.test_log_loss = None

    def train_and_predict(self, kernel):
        # self.decision_tree = Pipeline((
        #     ("scaler", StandardScaler()),
        #     ("random_forest", OneVsOneClassifier(DecisionTreeClassifier(random_state=42))),
        # ))

        self.decision_tree = Pipeline((
            ("scaler", StandardScaler()),
            ("decision_tree", DecisionTreeClassifier(random_state=42)),
        ))

        X, y = DataWrangler.get_data_and_target(self.train_data)
        self.decision_tree.fit(X, y)
        self.predict()

    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        print(self.decision_tree.score(X_train, y_train))
        print(self.decision_tree.score(X_test, y_test))


        result_test = self.decision_tree.predict(X_test)
        result_train = self.decision_tree.predict(X_train)
        print(numpy.array_equiv(result_train, y_train))
        print(numpy.array_equal(result_test, y_train))


        # self.train_error = mean_squared_error(y_train, train_result)
        # self.test_error = mean_squared_error(y_test, test_result)
        # self.train_log_loss = log_loss(y_train, train_result)
        # self.test_log_loss = log_loss(y_test, test_result)