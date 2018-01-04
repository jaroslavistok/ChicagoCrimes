from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVM:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self, kernel):
        svm_clf = Pipeline((
            ("scaler", StandardScaler()),
            ("linear_svc", SVC(kernel=kernel)),
        ))



        svm_clf.fit()





    def predict(self):
        pass

    def get_train_error(self):
        pass

    def get_train_log_loss(self):
        pass

    def get_test_error(self):
        pass

    def get_train_log_loss(self):
        pass