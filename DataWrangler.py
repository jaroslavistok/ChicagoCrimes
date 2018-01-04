import pandas


class DataWrangler:
    def __init__(self):
        self.crime_data = []
        self.census_data = []
        self.data = []

    def load_data(self, crime_data_filename, census_data_filename):
        self.crime_data = self._load_data_from_csv(crime_data_filename)
        self.census_data = self._load_data_from_csv(census_data_filename)
        self.data = self._join_data()

    def _load_data_from_csv(self, filename):
        return pandas.read_csv(filename)

    def _join_data(self):
        return self.crime_data.set_index('Community Area').join(self.census_data, lsuffix='_crime_data', rsuffix='_census_data')

    def prepare_data(self):
        self._parse_date()
        self._fill_missing_data()
        self._cast_data_to_correct_dtype()

    def merge_crime_categories(self, violent_crimes, non_violent_crimes):
        for index, row in self.data.iterrows():
            if row['Primary Type'] in violent_crimes:
                self.data.at[index, 'Primary Type'] = 1
            elif row['Primary Type'] in non_violent_crimes:
                self.data.at[index, 'Primary Type'] = 0
            else:
                self.data.at[index, 'Primary Type'] = 0

    def _parse_date(self):
        self.data['Date'] = pandas.to_datetime(self.data['Date'], format='%m/%d/%Y %I:%M:%S %p')

        dates = pandas.DataFrame({"year": self.data['Date'].dt.year,
                                  "month": self.data['Date'].dt.month,
                                  "day": self.data['Date'].dt.day,
                                  "hour": self.data['Date'].dt.hour,
                                  "dayofyear": self.data['Date'].dt.dayofyear,
                                  "week": self.data['Date'].dt.week,
                                  "weekofyear": self.data['Date'].dt.weekofyear,
                                  "dayofweek": self.data['Date'].dt.dayofweek,
                                  "weekday": self.data['Date'].dt.weekday,
                                  "quarter": self.data['Date'].dt.quarter,
                                  })

        self.data['year'] = dates['year']
        self.data['month'] = dates['month']
        self.data['day'] = dates['day']
        self.data['hour'] = dates['hour']
        self.data['dayofyear'] = dates['dayofyear']
        self.data['week'] = dates['week']
        self.data['weekofyear'] = dates['weekofyear']
        self.data['dayofweek'] = dates['dayofweek']
        self.data['weekday'] = dates['weekday']
        self.data['quarter'] = dates['quarter']
        self.data = self.data.drop("Date", axis=1)

    def _fill_missing_data(self):
        median = self.data['Latitude'].median()
        self.data['Latitude'].fillna(median, inplace=True)

        median = self.data['Longitude'].median()
        self.data['Longitude'].fillna(median, inplace=True)

        median = self.data['HARDSHIP INDEX'].median()
        self.data['HARDSHIP INDEX'].fillna(median, inplace=True)

    def _cast_data_to_correct_dtype(self):
        self.data['Arrest'] = self.data['Arrest'].astype(int)
        self.data['Domestic'] = self.data['Domestic'].astype(int)

    def drop_irrelevant_attributes(self, attributes_to_drop):
        self.data = self.data.drop(
           attributes_to_drop , axis=1)

        """"
        ["ID", "COMMUNITY AREA NAME", "Case Number", "Block", "IUCR", "Description", "Location Description",
         "Updated On", "Location", "X Coordinate", "Y Coordinate", "FBI Code", "Ward", "Year", "District",
         "Community Area Number"]
        """





