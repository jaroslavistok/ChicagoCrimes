from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Classifier import Classifier
from DataWrangler import DataWrangler

if __name__ == '__main__':
    dataWrangler = DataWrangler()
    dataWrangler.load_data('crimes2017.csv', 'census_data.csv')

    dataWranglerForMergedData = DataWrangler()
    dataWranglerForMergedData.load_data('crimes2017.csv', 'census_data.csv')

    size = 10000

    # dataWrangler.load_data('crimes_month.csv', 'census_data.csv')

    violent_crimes = ['BATTERY', 'ASSAULT', 'OTHER OFFENSE', 'OFFENSE INVOLVING CHILDREN', 'CRIM SEXUAL ASSAULT',
                      'SEX OFFENSE', 'HOMICIDE', 'KIDNAPPING', 'INTIMIDATION', 'OBSCENITY']

    non_violent_crimes = ['GAMBLING', 'STALKING', 'ARSON', 'NARCOTICS', 'THEFT', 'ROBBERY', 'DECEPTIVE PRACTICE',
                          'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT', 'BURGLARY', 'CRIMINAL TRESPASS',
                          'WEAPONS VIOLATION',
                          'PUBLIC PEACE VIOLATION', 'INTERFERENCE WITH PUBLIC OFFICER', 'PROSTITUTION',
                          'LIQUOR LAW VIOLATION',
                          'STALKING', 'CONCEALED CARRY LICENSE VIOLATION', 'PUBLIC INDECENCY', 'NON-CRIMINAL']

    irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
                             "Location Description",
                             "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
                             "District",
                             "Community Area Number"]

    # irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
    #                          "Location Description",
    #                          "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
    #                          "District",
    #                          "Community Area Number", 'Domestic', 'Latitude', 'Longitude', 'PERCENT OF HOUSING CROWDED',
    #                          'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
    #                          'PERCENT AGED UNDER 18 OR OVER 64', 'Beat', 'HARDSHIP INDEX', 'month',
    #                          'day', 'hour', 'dayofyear', 'week', 'weekofyear', 'dayofweek', 'weekday', 'quarter', 'Arrest']

    # irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description",
    #                          "Location Description",
    #                          "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year",
    #                          "District",
    #                          "Community Area Number", 'Beat', 'Latitude', 'Longitude', 'PERCENT OF HOUSING CROWDED',
    #                          'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED',
    #                          'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
    #                          'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME ', 'HARDSHIP INDEX', 'month',
    #                          'day', 'hour', 'dayofyear', 'week', 'weekofyear', 'dayofweek', 'weekday', 'quarter',
    #                          'Arrest']

    dataWrangler.prepare_data()
    dataWrangler.drop_irrelevant_attributes(irrelevant_attributes)
    dataWranglerForMergedData.prepare_data()
    dataWranglerForMergedData.drop_irrelevant_attributes(irrelevant_attributes)

    dataWranglerForMergedData.merge_crime_categories(violent_crimes, non_violent_crimes)
    dataWrangler.encode_categories_labels()

    # train_data, test_data = dataWrangler.split_data()
    train_data, test_data = dataWrangler.pick_samples_and_split_data(size)

    classifier = Classifier(train_data, test_data)

    #multiklasifikacia OneVsRest
    print("mlp multi")
    classifier.train_and_predict(MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=((100, 100)), alpha=1e-5, random_state=1))
    classifier.print_results()

    print("forest multi")
    classifier.train_and_predict(OneVsRestClassifier(RandomForestClassifier(random_state=42, n_estimators=100)))
    classifier.print_results()

    print("tree multi")
    classifier.train_and_predict(OneVsRestClassifier(DecisionTreeClassifier(random_state=42)))
    classifier.print_results()

    print('logistic reggresion multi')
    classifier.train_and_predict(OneVsRestClassifier(LogisticRegression(C=1e5)))
    classifier.print_results()

    print('svc rbf multi')
    classifier.train_and_predict(OneVsRestClassifier(SVC(kernel='rbf')))
    classifier.print_results()

    print('svc poly multi')
    classifier.train_and_predict(OneVsRestClassifier(SVC(kernel='poly')))
    classifier.print_results()



    # binarna klasifikacia

    train_data_merged, test_data_merged = dataWranglerForMergedData.pick_samples_and_split_data(size)

    classifier = Classifier(train_data_merged, test_data_merged)

    print("mlp")
    classifier.train_and_predict(MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=((50, 50)), alpha=1e-5, random_state=1))
    classifier.print_results()

    print("forest")
    classifier.train_and_predict(RandomForestClassifier(random_state=42, n_estimators=10))
    classifier.print_results()

    print("tree")
    classifier.train_and_predict(DecisionTreeClassifier(random_state=42))
    classifier.print_results()

    print('logistic reggresion')
    classifier.train_and_predict(LogisticRegression(C=1e5))
    classifier.print_results()

    print('svc rbf')
    classifier.train_and_predict(SVC(kernel='rbf'))
    classifier.print_results()

    print('svc poly')
    classifier.train_and_predict(SVC(kernel='poly'))
    classifier.print_results()

