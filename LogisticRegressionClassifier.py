from sklearn.linear_model import LogisticRegression
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

    def train_and_predict(self, kernel):
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

        print(self.logistic_regression.score(X_train, y_train))
        print(self.logistic_regression.score(X_test, y_test))

        # print(self.logistic_regression.score(X_train, y_train))
        # print(self.logistic_regression.score(X_test, y_test))



        # self.train_error = mean_squared_error(y_train, train_result)
        # self.test_error = mean_squared_error(y_test, test_result)
        # self.train_log_loss = log_loss(y_train, train_result)
        # self.test_log_loss = log_loss(y_test, test_result)