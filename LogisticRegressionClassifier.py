from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from DataWrangler import DataWrangler


class LogisticRegressionClassifier:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.logistic_regression = None

        self.train_error = None
        self.test_error = None
        self.train_log_loss = None
        self.test_log_loss = None

    def train_and_predict(self):
        self.logistic_regression = Pipeline((
            ("scaler", StandardScaler()),
            ("random_forest", LogisticRegression(C=1e5)),
        ))

        X, y = DataWrangler.get_data_and_target(self.train_data)
        self.logistic_regression.fit(X, y)
        self.predict()

    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        print(accuracy_score(y_train, self.logistic_regression.predict(X_train)))
        print(accuracy_score(y_test, self.logistic_regression.predict(X_test)))

        print(cross_val_score(self.logistic_regression, X_train, y_train, cv=10))
        print(cross_val_score(self.logistic_regression, X_test, y_test, cv=10))



        # self.train_error = mean_squared_error(y_train, train_result)
        # self.test_error = mean_squared_error(y_test, test_result)
        # self.train_log_loss = log_loss(y_train, train_result)
        # self.test_log_loss = log_loss(y_test, test_result)