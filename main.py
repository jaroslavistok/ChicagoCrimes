from sklearn.decomposition import PCA
from DataWrangler import DataWrangler
from DecisionTree import DecisionTree
from LogisticRegressionClassifier import LogisticRegressionClassifier
from MLP import MLP
from RandomForest import RandomForest
from SVM import SVM
import numpy as np
from matplotlib import pyplot as plt


"""
42433 training samples (70%)
18186 test samples (30%)
"""

if __name__ == '__main__':
    dataWrangler = DataWrangler()
    # dataWrangler.load_data('crimes2017.csv', 'census_data.csv')
    dataWrangler.load_data('crimes2017.csv', 'census_data.csv')


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

    # irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
    #                          "Location Description",
    #                          "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
    #                          "District",
    #                          "Community Area Number", 'Domestic', 'Latitude', 'Longitude', 'PERCENT OF HOUSING CROWDED',
    #                          'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
    #                          'PERCENT AGED UNDER 18 OR OVER 64', 'Beat', 'HARDSHIP INDEX', 'month',
    #                          'day', 'hour', 'dayofyear', 'week', 'weekofyear', 'dayofweek', 'weekday', 'quarter', 'Arrest']

    irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
                             "Location Description",
                             "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
                             "District",
                             "Community Area Number", 'Beat', 'Latitude', 'Longitude', 'PERCENT OF HOUSING CROWDED',
                             'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED',
                             'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
                             'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME ', 'HARDSHIP INDEX', 'month',
                             'day', 'hour', 'dayofyear', 'week', 'weekofyear', 'dayofweek', 'weekday', 'quarter',
                             'Arrest']

    dataWrangler.prepare_data()
    dataWrangler.drop_irrelevant_attributes(irrelevant_attributes)

    dataWrangler.merge_crime_categories(violent_crimes, non_violent_crimes)
    print(dataWrangler.data.info())
    # dataWrangler.encode_categories_labels()

    # print(dataWrangler.data.info()

    # print(dataWrangler.data['month'])
    # plot.show()
    # exit()

    train_data, test_data = dataWrangler.split_data()
    """
    svm = SVM(train_data, test_data)
    print('rbf')
    svm.train_and_predict('rbf')
    print('poly')
    svm.train_and_predict('rbf')
    """


    random_forest = RandomForest(train_data, test_data)
    random_forest.train_and_predict()

    """
    decision_tree = DecisionTree(train_data, test_data)
    decision_tree.train_and_predict()
    """
    """
    mlp = MLP(train_data, test_data)
    mlp.train_and_predict()
    """
    """
    logistic_regression = LogisticRegressionClassifier(train_data, test_data)
    logistic_regression.train_and_predict()
    """

