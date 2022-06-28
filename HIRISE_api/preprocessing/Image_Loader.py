import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

class Load_Hirise_Images(Dataset):
    """Hirise Image dataset."""
    def __init__(self,
                 path_to_images,
                 transform=None,
                 train=True):
        # ------------------------------------------------------------------------------
        # path_to_images: where you put the image dataset
        # transform:  data transform
        # img_size: resize all images to a standard size
        # train: return training set or test set
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
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, lengths)

    def __len__(self):
        ...
    def __getitem__(self, idx):
        ...