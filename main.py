from DataWrangler import DataWrangler
from SVM import SVM

"""
42433 training samples (70%)
18186 test samples (30%)

"""


if __name__ == '__main__':
    dataWrangler = DataWrangler()
    dataWrangler.load_data('crimes.csv', 'census_data.csv')

    violent_crimes = ['BATTERY', 'ASSAULT', 'OTHER OFFENSE', 'OTHER OFFENSE', 'OFFENSE INVOLVING CHILDREN', 'CRIM SEXUAL ASSAULT',
                 'SEX OFFENSE', 'HOMICIDE', 'KIDNAPPING', 'INTIMIDATION', 'OBSCENITY', 'NON-CRIMINAL']
    non_violent_crimes = ['GAMBLING', 'STALKING', 'ARSON', 'NARCOTICS', 'THEFT', 'ROBBERY', 'DECEPTIVE PRACTICE',
                     'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT', 'BURGLARY', 'CRIMINAL TRESPASS', 'WEAPONS VIOLATION',
                     'PUBLIC PEACE VIOLATION','INTERFERENCE WITH PUBLIC OFFICER', 'PROSTITUTION', 'LIQUOR LAW VIOLATION',
                     'STALKING', 'CONCEALED CARRY LICENSE VIOLATION', 'PUBLIC INDECENCY']

    irrelevant_attributes = ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description", "Location Description",
         "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year", "District",
         "Community Area Number"]

    dataWrangler.drop_irrelevant_attributes(irrelevant_attributes)
    dataWrangler.prepare_data()
    dataWrangler.merge_crime_categories(violent_crimes, non_violent_crimes)
    train_data, test_data = dataWrangler.split_data()

    svm = SVM(train_data, test_data)
    svm.train_data('linear')





