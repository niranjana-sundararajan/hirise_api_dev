from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

import os
import numpy as np
import torch
import warnings

if __package__ is None or __package__ == "":
    import Data_Preparation
    import Encoding
else:
    from . import Data_Preparation
    from . import Encoding

warnings.filterwarnings("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))


class HiriseImageDataset(Dataset):
    """Hirise Image Dataset Class that initialize the pytorch ImageLoader
    Dataset
    with the folder images to return and image and associated folder  name(
    label)
    """

    def __init__(self, path_to_images, transform=None):
        # -------------------------------------------------------------------
        # path_to_images: where you put the image dataset
        # transform:  data transform
        # img_size: resize all images to a standard size
        # -------------------------------------------------------------------

        # Load all the images and their labels
        self.dataset = datasets.ImageFolder(
            path_to_images, transform=transform
        )
        self.len = len(self.dataset.imgs)
        self.path_to_images = path_to_images

        # -------------------------------------------------------------------
        # Split the data into train and test data 80 : 20
        # -------------------------------------------------------------------
        # Calculate the lengths of the vectors
        lengths = [
            int(np.ceil(len(self.dataset) * 0.8)),
            int(np.floor(len(self.dataset) * 0.2)),
        ]

        # Extract the images and labels
        self.train_dataset, self.test_dataset = random_split(
            self.dataset, lengths
        )

    def __len__(self):
        # Return the number of samples
        return self.len

    def __getitem__(self, idx):
        # Get each item and target of the image loader dataset
        sample, target = self.data[idx], self.data[idx]
        sample = sample.view(1, 256, 256).float() / 255.0
        if self.transform:
            sample = self.transform(sample)
            target = self.transform(target)
        return sample, target


def generate_dataset(folder_path, transform=None):
    """
    Function that generated the HIRISE Dataset given a folderpath of HIRISE
    Images
    """
    # Define the transformation function for the dataset
    if not transform:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(0.40655, 0.1159),
                transforms.Grayscale(num_output_channels=1),
            ]
        )
    # Initialize the data prep class
    dp = Data_Preparation.DataPreparation()

    # Get image dataset
    dataset = dp.get_image_dataset(
        f_path=folder_path, transform_data=transform
    )
    return dataset


def initialize_encoder_decoder(latent_dimensions=2000):
    """
    Fuction that initialized the encoder and decoder depeining on the latent
     dimensions specified by the user.
    Default is 2000 dimenions.
    """
    # Initialzie the encoder and decoder from the Encoding Module
    encoder = Encoding.CAEEncoder(
        encoded_space_dim=latent_dimensions, fc2_input_dim=256
    )
    decoder = Encoding.CAEDecoder(
        encoded_space_dim=latent_dimensions, fc2_input_dim=256
    )

    # Check if the GPU is available
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Move both the encoder and the decoder to the selected device
    return encoder.to(device), decoder.to(device)


def generate_dataloaders(folder_path, transform=None):
    """
    Function that generates the dataloaders for a HIRISE dataset, given folder
    path specified by the user
    """

    # Initialize the data preparation class
    dp = Data_Preparation.DataPreparation()

    # Define the transforms if not present
    if not transform:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(0.40655, 0.1159),
                transforms.Grayscale(num_output_channels=1),
            ]
        )

    # Generate the dataset
    datasets = generate_dataset(folder_path=folder_path, transform=transform)

    # Get the training, test and validation tensors
    tr, tst, val = dp.get_train_test_val_tensors(dataset=datasets)

    # Get the required dataloaders
    train_loader, test_loader, val_loader = dp.get_train_test_val_dataloader(
        tr, tst, val
    )
    return train_loader, test_loader, val_loader


def show_encoder_decoder_image_sizes(
    folder_path, device="cpu", transform=None
):
    """
    Function that returns the input and output image sizes of images that have
    been through the autoencoding process
    """
    # Generate the ImageLoader dataset
    datasets = generate_dataset(folder_path=folder_path, transform=transform)

    # Select first image from the dataset
    img, _ = datasets.train_dataset[0]

    # unsqueeze from the dataloader "batch" format(Add the batch dimension
    # in  the first axis)
    img = img.unsqueeze(0).to(device)

    # Print the original shape
    print("Original image shape:", img.shape)
    encoder, decoder = initialize_encoder_decoder(latent_dimensions=2000)

    # Encode the image
    img_enc = encoder(img)

    # Print the shape after encoding
    print("Encoded image shape:", img_enc.shape)


def show_classes(folder_path, transform=None, dict_values=True):
    """
    Function that shows all classes defined by the user though the Image
    Folders using the Image Folder dataset
    """
    # Generate the ImageLoader dataset
    datasets = generate_dataset(folder_path=folder_path, transform=transform)

    if dict_values:
        return datasets.dataset.class_to_idx
    else:
        return datasets.dataset.classes
