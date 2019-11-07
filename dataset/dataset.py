# This file is for different Dataset loading
from torch.utils.data import Dataset
import pandas as pd
from skimage import io

class Dataset(Dataset):
    def __init__(self, root, train, split, transform):
        # Define parameters for dataset
        self.train = train
        self.split = split
        self.transform = transform

        # Define the dataset path for future loading
        self.image_root = root + 'Twitter1/Images/'
        self.spilt_root = root + 'Twitter1/Split/'
        self.split_train_path = self.spilt_root + 'train_' + str(split) + '.txt'
        self.split_test_path = self.spilt_root + 'test_' + str(split) + '.txt'

        # Read training and test data from the defined train & split path
        self.train_dataframe = pd.read_csv(self.split_train_path, sep = " ", names = ['image_path','label'])
        self.test_dataframe = pd.read_csv(self.split_test_path, sep = " ", names = ['image_path','label'])

        # Check dataset lens
        self.train_data_length =  self.train_dataframe.shape[0]
        self.test_data_lenght = self.test_dataframe.shape[0]


    def __getitem__(self, index):
        # If is training, then read the training dataset and vice verse
        if self.train is True:
            image = io.imread(self.image_root + self.train_dataframe['image_path'][index])
            image_tensor = self.transform(image)
            label = self.train_dataframe['label'][index]
        else:
            image = io.imread(self.image_root + self.test_dataframe['image_path'][index])
            image_tensor = self.transform(image)
            label = self.train_dataframe['label'][index]

        return image_tensor, label

    def __len__(self):
        if self.train is True:
            return self.train_data_length
        else:
            return self.test_data_lenght


