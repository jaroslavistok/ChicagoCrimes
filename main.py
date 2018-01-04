from DataWrangler import DataWrangler

if __name__ == '__main__':
    dataWrangler = DataWrangler()
    dataWrangler.load_data('crime.csv', 'census_data.csv')

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
    print(len(train_data))
    print(len(test_data))



