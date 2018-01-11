from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
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

    def train_and_predict(self):
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

        print(accuracy_score(y_train, self.mlp.predict(X_train)))
        print(accuracy_score(y_test, self.mlp.predict(X_test)))

        print(cross_val_score(self.mlp, X_train, y_train, cv=10))
        print(cross_val_score(self.mlp, X_test, y_test, cv=10))
