import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Import torch packages that help us define our network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Normalize
from torchvision import datasets, transforms, models

# Package that allows us to summarize our network
from torchsummary import summary


import warnings
warnings.filterwarnings("ignore")

class Hirise_Image_Dataset(Dataset):
    """Hirise Image dataset."""
    def __init__(self,
                 path_to_images,
                 transform=None):
        # ------------------------------------------------------------------------------
        # path_to_images: where you put the image dataset
        # transform:  data transform
        # img_size: resize all images to a standard size
        # ------------------------------------------------------------------------------

        # Load all the images and their labels
        self.dataset = datasets.ImageFolder(path_to_images, transform=transform)
        self.len = len(self.dataset.imgs)
        self.path_to_images = path_to_images

        # ------------------------------------------------------------------------------
        # Split the data into train and test data 80 : 20
        # ------------------------------------------------------------------------------
        # Calculate the lengths of the vectors
        lengths = [int(np.ceil(len(self.dataset)*0.8)), int(np.floor(len(self.dataset)*0.2))]


        # Extract the images and labels   
        self.train_dataset, self.test_dataset = random_split(self.dataset, lengths)

    def __len__(self):
        # Return the number of samples
        return self.len

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample