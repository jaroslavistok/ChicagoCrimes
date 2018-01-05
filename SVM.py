from sklearn.metrics import mean_squared_error, log_loss
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
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
        # self.svm = Pipeline((
        #     ("scaler", StandardScaler()),
        #     ("linear_svc", SVC(kernel=kernel)),
        # ))

        self.svm = Pipeline((
            ("scaler", StandardScaler()),
            ("linear_svc", SVC(kernel='poly', degree=2, coef0=1, C=5)),
        ))

        X, y = DataWrangler.get_data_and_target(self.train_data)
        self.svm.fit(X, y)
        self.predict()

    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        print(self.svm.score(X_train, y_train))
        print(self.svm.score(X_test, y_test))

        # train_result = self.svm.predict(X_train)
        # test_result = self.svm.predict(X_test)

        # self.train_error = mean_squared_error(y_train, train_result)
        # self.test_error = mean_squared_error(y_test, test_result)
        # self.train_log_loss = log_loss(y_train, train_result)
        # self.test_log_loss = log_loss(y_test, test_result)

