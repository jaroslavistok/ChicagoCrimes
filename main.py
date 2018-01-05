from DataWrangler import DataWrangler
from DecisionTree import DecisionTree
from LogisticRegressionClassifier import LogisticRegressionClassifier
from MLP import MLP
from RandomForest import RandomForest
from SVM import SVM

"""
42433 training samples (70%)
18186 test samples (30%)

"""


if __name__ == '__main__':
    dataWrangler = DataWrangler()
    dataWrangler.load_data('crimes_month.csv', 'census_data.csv')

    violent_crimes = ['BATTERY', 'ASSAULT', 'OTHER OFFENSE', 'OFFENSE INVOLVING CHILDREN', 'CRIM SEXUAL ASSAULT',
                      'SEX OFFENSE', 'HOMICIDE', 'KIDNAPPING', 'INTIMIDATION', 'OBSCENITY']

    non_violent_crimes = ['GAMBLING', 'STALKING', 'ARSON', 'NARCOTICS', 'THEFT', 'ROBBERY', 'DECEPTIVE PRACTICE',
                          'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT', 'BURGLARY', 'CRIMINAL TRESPASS',
                          'WEAPONS VIOLATION',
                          'PUBLIC PEACE VIOLATION', 'INTERFERENCE WITH PUBLIC OFFICER', 'PROSTITUTION',
                          'LIQUOR LAW VIOLATION',
                          'STALKING', 'CONCEALED CARRY LICENSE VIOLATION', 'PUBLIC INDECENCY', 'NON-CRIMINAL']

    # irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
    #                          "Location Description",
    #                          "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
    #                          "District",
    #                          "Community Area Number"]



    irrelevant_attributes = []

    dataWrangler.drop_irrelevant_attributes(irrelevant_attributes)
    dataWrangler.prepare_data()
    # dataWrangler.merge_crime_categories(violent_crimes, non_violent_crimes)
    dataWrangler.encode_categories_labels()
    dataWrangler.encode_non_numerical_values(["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
                               "Location Description",
                               "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
                               "District",
                               "Community Area Number"])

    # dataWrangler.data = dataWrangler.data.drop(['Arrest', 'Domestic', 'Beat', 'Latitude', "Longitude", "month", "day", "hour", "week",
    #                         "dayofweek", "dayofyear", "weekday", "quarter", "weekofyear", 'HARDSHIP INDEX', 'PERCENT OF HOUSING CROWDED',
    #                         'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
    #                                             'PERCENT AGED UNDER 18 OR OVER 64'], axis=1)
    # print(dataWrangler.data.info())
    train_data, test_data = dataWrangler.split_data()

    """
    svm = SVM(train_data, test_data)
    svm.train_and_predict('linear')
    print("Train error: " + str(svm.train_error))
    print("Test error: " + str(svm.test_error))
    print("Train log loss: " + str(svm.train_log_loss))
    print("Test log loss: " + str(svm.test_log_loss))
    """

    random_forest = RandomForest(train_data, test_data)
    random_forest.train_and_predict('linear')
    print("Train error: " + str(random_forest.train_error))
    print("Test error: " + str(random_forest.test_error))
    print("Train log loss: " + str(random_forest.train_log_loss))
    print("Test log loss: " + str(random_forest.test_log_loss))

    """
    decision_tree = DecisionTree(train_data, test_data)
    decision_tree.train_and_predict('linear')
    print("Train error: " + str(decision_tree.train_error))
    print("Test error: " + str(decision_tree.test_error))
    print("Train log loss: " + str(decision_tree.train_log_loss))
    print("Test log loss: " + str(decision_tree.test_log_loss))
    """
    """
    mlp = MLP(train_data, test_data)
    mlp.train_and_predict('linear')
    print("Train error: " + str(mlp.train_error))
    print("Test error: " + str(mlp.test_error))
    print("Train log loss: " + str(mlp.train_log_loss))
    print("Test log loss: " + str(mlp.test_log_loss))
    """

    """
    logistic_regression = LogisticRegressionClassifier(train_data, test_data)
    logistic_regression.train_and_predict('linear')
    print("Train error: " + str(logistic_regression.train_error))
    print("Test error: " + str(logistic_regression.test_error))
    print("Train log loss: " + str(logistic_regression.train_log_loss))
    print("Test log loss: " + str(logistic_regression.test_log_loss))
    """

