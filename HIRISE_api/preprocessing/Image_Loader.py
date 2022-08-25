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

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from hirise import Image_Client
    import Data_Preparation
else:
    from . import Data_Preparation
# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
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
        sample, target = self.data[idx], self.data[idx]
        sample = sample.view(1, 256, 256).float()/255.
        if self.transform:
            sample = self.transform(sample)
            target = self.transform(target)
        return sample, target


def generate_dataset(folder_path, transform = None):
        if not transform:
         transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)),transforms.Normalize(0.40655,0.1159), transforms.Grayscale(num_output_channels=1)])
        dp = Data_Preparation()
        dataset = dp.get_image_dataset(f_path = folder_path, transform_data = transform)
        return dataset

def initialize_encoder_decoder(latent_dimensions = 2000):
    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = CAE_Encoder(encoded_space_dim=latent_dimensions,fc2_input_dim=256)
    decoder = CAE_Decoder(encoded_space_dim=latent_dimensions,fc2_input_dim=256)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Move both the encoder and the decoder to the selected device
    return encoder.to(device), decoder.to(device)


def generate_dataloaders(folder_path , transform = None):
    dp = Data_Preparation()
    if not transform:
        transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)),transforms.Normalize(0.40655,0.1159), transforms.Grayscale(num_output_channels=1)])
    datasets = generate_dataset(folder_path = folder_path, transform = transform)
    tr,tst,val = dp.get_train_test_val_tensors(dataset = datasets)
    train_loader,test_loader, val_loader = dp.get_train_test_val_dataloader(tr,tst,val)
    return train_loader,test_loader, val_loader

def show_encoder_decoder_image_sizes(folder_path , transform = None):
    datasets = generate_dataset(folder_path = folder_path, transform = transform)
    img, _ = datasets.train_dataset[0]
    img = img.unsqueeze(0).to(device) # Add the batch dimension in the first axis
    print('Original image shape:', img.shape)
    encoder,decoder = initialize_encoder_decoder(latent_dimensions = 2000)
    img_enc = encoder(img)
    print('Encoded image shape:', img_enc.shape)



def show_classes(folder_path , transform = None, dict_values = True):
    datasets = generate_dataset(folder_path =folder_path, transform = transform)
        if dict_values:
            return datasets.dataset.class_to_idx
        else:
            return datasets.dataset.classes