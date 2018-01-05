from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from DataWrangler import DataWrangler


class MLP:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.mlp = None

        self.train_error = None
        self.test_error = None
        self.train_log_loss = None
        self.test_log_loss = None

    def train_and_predict(self, kernel):
        self.mlp = Pipeline((
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=((100, 100)), alpha=1e-5, random_state=1)),
        ))



        X, y = DataWrangler.get_data_and_target(self.train_data)
        self.mlp.fit(X, y)
        self.predict()

    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        train_result = self.mlp.predict(X_train)
        test_result = self.mlp.predict(X_test)
        print(self.mlp.score(X_train, y_train))
        print(self.mlp.score(X_test, y_test))



        # print(y_train)
        # print(train_result)


        self.train_error = mean_squared_error(y_train, train_result)
        self.test_error = mean_squared_error(y_test, test_result)
        # self.train_log_loss = log_loss(y_train, train_result)
        # self.test_log_loss = log_loss(y_test, test_result)