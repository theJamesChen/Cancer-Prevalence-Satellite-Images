import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import numpy as np
from io import BytesIO
import argparse
from PIL import Image
from urllib import request
import matplotlib.pyplot as plt # this is if you want to plot the map using pyplot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import os

parser = argparse.ArgumentParser(description='Deep Learning JHU Final Project')
parser.add_argument('--weights', type=str, default='', metavar='W',
                    help='Load weights file.')


# Get the map region using Google Static Maps.
def get_area_map(latitude, longitude):
    url = "https://maps.googleapis.com/maps/api/staticmap?center=%d,%d&zoom=14&size=224x224&maptype=satellite&key=AIzaSyBWZgOcVdjJs-TsmRlN_O7m0tZdxlhQHzM" % (latitude, longitude)
    img_array = np.asarray(bytearray(request.urlopen(url).read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, 1)
    return image

# Parse a csv file of data.
def parse_csv_file(csv_file):
    data = pd.read_csv(csv_file)
    return data



class MapDataSet(Dataset):
    """Map area dataset."""

    def __init__(self, df, transform=None, data_aug=False):
        """
        Args:
            df (dataframe): Dataframe with city names, data values, and geolocations.
            transform: transforms required
            data_aug: Data augmentation or not.
        """
        self.df = df
        self.transform=transform
        self.data_aug=data_aug

    # Returns length of the data set.
    def __len__(self):
            return(self.df.shape[0])

    # Gets a specific item from the data set.
    def __getitem__(self, index):
        coords = self.df.iloc[index]['GeoLocation']
        coords = coords.translate(str.maketrans('','','() '))
        path = os.path.join(os.getcwd(), 'images/')
        image = cv2.imread(os.path.join(path ,coords+'.png'), 1)
        label = int(self.df.iloc[index]['Data_Category'])
        sample = {'image' : image, 'label' : label}

        if self.transform:
            image = sample['image']
            image = self.transform(image)
            sample['image'] = image

        return sample

class ModifiedVGG(nn.Module):
    """Neural Net Module."""
    def __init__(self, pretrained_model):
        super(ModifiedVGG, self).__init__()
        self.pretrained_model = pretrained_model
        #self.second_last_layer = nn.Linear(1000, 100)
        self.last_layer = nn.Linear(1000, 1)

    def forward(self, x):
        #return self.last_layer(self.second_last_layer(self.pretrained_model(x)))
        return self.last_layer(self.pretrained_model(x))


def train(train_loader, test_loader, data_aug=False):

    loss_func = nn.MSELoss().cuda()
    # learning_rate = 1e-6
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs=10
    i = 0
    j = 0
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            image = data['image'].cuda()
            label = (data['label']).float().cuda()

            optimizer.zero_grad()

            output = model(image)
            output = torch.squeeze(output)
            loss = loss_func(output, label)

            # Backward pass
            loss.backward(retain_graph=True)
            train_loss.append(loss.data[0])
            optimizer.step()

            i += 1
            print("Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, loss.data[0]))

        for batch_idx, data in enumerate(test_loader):
            image = data['image'].cuda()
            label = (data['label']).float().cuda()

            output = model(image)
            output = torch.squeeze(output)
            test_loss_temp = F.mse_loss(output, label)
            test_loss.append(test_loss_temp.data[0])
            j = j+1

            print("Batch %d Loss %f" % (batch_idx, loss.data[0]))

    plt.figure()
    iterations = range(1, i+1)
    plt.plot(iterations, train_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('train_loss_2layer_20ep.png')

    plt.figure()
    iterations = range(1, j+1)
    plt.plot(iterations, test_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('test_loss_2layer_20ep.png')

def test(data_loader, data_aug=False):
    count = 0
    correct = 0
    for batch_idx, data in enumerate(data_loader):
        image = data['image'].cuda()
        label = (data['label']).float().numpy()
        output = model(image)
        output = np.squeeze(torch.round(output).detach().cpu().numpy())
        correct = correct  + np.sum(np.equal(output, label))
        count = batch_idx * 256
    print(correct/count)


#         output = torch.squeeze(output)
#         loss = F.l1_loss(output, label)
        

#         print("Batch %d Loss %f" % (batch_idx, loss.data[0]))



N=256

num_categories = 6
data = parse_csv_file('500_Cities__Local_Data_for_Better_Health__2017_release.csv')
data_value = data[['CityName', 'Data_Value', 'GeoLocation']]
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

print('Splitting into train and test datasets...')
# Split into train and test data.
train_df = data_value.sample(frac=0.8)
test_df = data_value.drop(train_df.index)

print('Creating the datasets...')
#Create training and test datasets.
trans = transforms.Compose([transforms.ToTensor()])
train_set = MapDataSet(df=train_df, transform=trans)
test_set = MapDataSet(df=test_df, transform=trans)

print('Initializing dataloaders...')
# Create data loades for train and test datsets.
train_loader = DataLoader(dataset=train_set, batch_size=N, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_set, batch_size=N, shuffle=False, num_workers=2)



pretrained_model = torchvision.models.vgg16(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False
model = ModifiedVGG(pretrained_model).cuda()

params = list(model.last_layer.parameters())
#params = list(model.last_layer.parameters()) + list(model.second_last_layer.parameters())
optimizer = optim.Adam(params, lr=1e-6)


# model = models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(1000, 1)



if __name__ == '__main__':
    args = parser.parse_args()
    if not args.weights:
        train(train_loader, test_loader)
        torch.save(model.state_dict(), 'weight_file.txt')
    else:
        weight_file = args.weights
        model.load_state_dict(torch.load(weight_file))
        test(test_loader)



