import pandas as pd
import numpy as np



def parse_csv_file(csv_file):
    data = pd.read_csv(csv_file)
    return data

data = parse_csv_file('500_Cities__Local_Data_for_Better_Health__2017_release.csv')
data_value = data[['CityName', 'Data_Value', 'PopulationCount', 'GeoLocation']]

print(data_value.loc['CityName'])