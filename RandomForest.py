
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MaxAbsScaler
from matplotlib import pyplot as plot

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve
import numpy as np

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

    def train_and_predict(self):
        self.random_forest = Pipeline((
            ("scaler", StandardScaler()),
            # ("random_forest", OneVsRestClassifier(RandomForestClassifier(random_state=42, n_estimators=300))),
            ("random_forest", RandomForestClassifier(random_state=42, n_estimators=20)),
        ))

        X, y = DataWrangler.get_data_and_target(self.train_data)



        self.random_forest.fit(X, y)
        self.predict()

    def predict(self):
        X_train, y_train = DataWrangler.get_data_and_target(self.train_data)
        X_test, y_test = DataWrangler.get_data_and_target(self.test_data)

        # plot.matshow(self.train_data.corr())
        # plot.show()
        # correlations = self.train_data.corr()
        #plot correlation matrix
        # fig = plot.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(correlations, vmin=-1, vmax=1)
        # fig.colorbar(cax)
        # ticks = np.arange(0, 21, 1)
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        # fig.show()

        # n_classes = y_train.shape[0]

        # y_probabilities = cross_val_predict(self.random_forest, X_test, y_test, cv=3, method="predict_proba")

        # y_scores_forest = y_probabilities[:, 1]
        # fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_scores_forest)
        # print(tpr_forest)
        # print(fpr_forest)
        # plot.plot(fpr_forest, tpr_forest, linewidth=10, label="Random forest")
        # plot.plot([0, 1], [0, 1], "k--")
        # plot.axis([0, 1, 0, 1])
        # plot.xlabel("False positive rate")
        # plot.ylabel("True positive rate")
        # plot.show()


        print(accuracy_score(y_train, self.random_forest.predict(X_train)))
        print(accuracy_score(y_test, self.random_forest.predict(X_test)))

        print(cross_val_score(self.random_forest, X_train, y_train, cv=10))
        print(cross_val_score(self.random_forest, X_test, y_test, cv=10))