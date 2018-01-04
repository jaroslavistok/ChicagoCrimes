from sklearn.metrics import mean_squared_error, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from DataWrangler import DataWrangler


class SVM:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.svm = None

        self.train_error = None
        self.test_error = None
        self.train_log_loss = None
        self.test_log_loss = None

    def train_and_predict(self, kernel):
        self.svm = Pipeline((
            ("scaler", StandardScaler()),
            ("linear_svc", SVC(kernel=kernel)),
        ))

        X, y = DataWrangler.get_data_and_target(self.train_data)
        self.svm.fit(X, y)
        self.svm.predict()


    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        train_result = self.svm.predict(X_train)
        test_result = self.svm.predict(X_test)

        self.train_error = mean_squared_error(y_train, train_result)
        self.test_error = mean_squared_error(y_test, test_result)
        self.train_log_loss = log_loss(y_train, train_result)
        self.test_log_loss = log_loss(y_test, test_result)


    def get_train_error(self):
        pass

    def get_train_log_loss(self):
        pass

    def get_test_error(self):
        pass

    def get_train_log_loss(self):
        pass