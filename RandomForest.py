from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MaxAbsScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from DataWrangler import DataWrangler


class RandomForest:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.random_forest = None

        self.train_error = None
        self.test_error = None
        self.train_log_loss = None
        self.test_log_loss = None

    def train_and_predict(self, kernel):
        self.random_forest = Pipeline((
            ("scaler", StandardScaler()),
            ("random_forest", OneVsOneClassifier(RandomForestClassifier(random_state=42))),
        ))

        X, y = DataWrangler.get_data_and_target(self.train_data)
        self.random_forest.fit(X, y)
        self.predict()

    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        # print(self.random_forest.score(X_train, y_train))
        # print(self.random_forest.score(X_test, y_test))

        print(accuracy_score(y_train, self.random_forest.predict(X_train)))
        print(accuracy_score(y_test, self.random_forest.predict(X_test)))


        # self.train_error = mean_squared_error(y_train, train_result)
        # self.test_error = mean_squared_error(y_test, test_result)
        # self.train_log_loss = log_loss(y_train, train_result)
        # self.test_log_loss = log_loss(y_test, test_result)