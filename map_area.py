from io import BytesIO
from PIL import Image
from urllib import request
import pandas as pd
import numpy as np
import cv2
import os


# Parse a csv file of data.
def parse_csv_file(csv_file):
    data = pd.read_csv(csv_file)
    return data

# Get the map region using Google Static Maps.
def get_area_map(latitude, longitude):
    url = "https://maps.googleapis.com/maps/api/staticmap?center=%f,%f&zoom=14&size=224x224&maptype=satellite&key=" % (latitude, longitude)
    img_array = np.asarray(bytearray(request.urlopen(url).read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, 1)
    cv2.imwrite(os.path.join(path ,str(latitude)+','+ str(longitude)+'.png'), image)




data = parse_csv_file('500_Cities__Local_Data_for_Better_Health__2017_release.csv')
data_value = data[['Data_Value','GeoLocation']]
data_value = data_value[np.isfinite(data_value['Data_Value'])]
min_val = data_value['Data_Value'].min()
max_val = data_value['Data_Value'].max()
increment = (max_val - min_val)/6


conditions = [
    (data_value['Data_Value'] >= min_val) & (data_value['Data_Value'] < (min_val+increment)),
    (data_value['Data_Value'] >= (min_val+increment)) & (data_value['Data_Value'] < (min_val+(2*increment))),
    (data_value['Data_Value'] >= (min_val+increment)) & (data_value['Data_Value'] < (min_val+(3*increment))),
    (data_value['Data_Value'] >= (min_val+increment)) & (data_value['Data_Value'] < (min_val+(4*increment))),
    (data_value['Data_Value'] >= (min_val+increment)) & (data_value['Data_Value'] < (min_val+(5*increment))),
    (data_value['Data_Value'] >= (min_val+increment))
    ]
choices = [1,2,3,4,5,6]
data_value['Data_Category'] = np.select(conditions, choices, default='')

print(data_value)

# for i in range(6):
    # max_temp_val = min_val + increment
    # data_value['Data_Category'] = np.where((data_value['Data_Value']>=min_val & data_value['Data_Value']<(max_temp_val)), 1, 0)
#     # print(min_val)
#     # print(min_val+increment)
#     #data_value['Data_Category'] = np.where(data_value['Data_Value']>=min_val & data_value['Data_Value']<(min_val+increment), i, '')
#     conditions = [
#     (data_value['Data_Value'] >= min_val) & (data_value['Data_Value'] < (min_val+increment))]
#     choices = [i]
#     data_value['Data_Category'] = np.select(conditions, choices, default='')
#     min_val = min_val + increment



# path = os.path.join(os.getcwd(), 'images/')
# i = 1
# for index, row in data_value.iterrows():
#     print(i)
#     coords = row['GeoLocation']
#     coords = coords.translate(str.maketrans('','','() ')).split(',')
#     get_area_map(float(coords[0]), float(coords[1]))
#     i = i + 1
