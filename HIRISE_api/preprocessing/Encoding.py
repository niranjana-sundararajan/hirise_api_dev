from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.xception import Xception
from keras.models import Model
from tensorflow.keras.utils import img_to_array
from torchvision import transforms
from tqdm import tqdm

from hirise.Image_Client import ImageClient

if __package__ is None or __package__ == '':
    import Image_Loader
    import utils
else:
    from . import Image_Loader
    from . import utils

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

os.environ['OPEgrid_columnsV_IO_ENABLE_JASPER'] = 'true'
device = torch.device("cuda") if torch.cuda.is_available() else \
    torch.device("cpu")  # Check if the GPU is available


class CAEEncoder(nn.Module):
    '''
    Class that supports functions needed to define the architecture and
    forward functions of the encoder in the Convolutional Autoencoder
    '''

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.ImageClient = ImageClient()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(31 * 31 * 32, 1024),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(1024, encoded_space_dim)
        )

    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # Apply linear layers
        x = self.encoder_lin(x)
        return x


class CAEDecoder(nn.Module):
    '''
    Class that supports functions needed to define the architecture and
    forward functions of the decoder in the Convolutional Autoencoder
    '''

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        # Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 1024),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(1024, 31 * 31 * 32),
            nn.ReLU(True)
        )

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 31, 31))

        # Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                               output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1,
                               output_padding=1)
        )

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1  (valid
        # pixel values)
        x = torch.sigmoid(x)
        return x


def train_CAE(encoder, decoder, device, dataloader, loss_fn, optimizer):
    """
    Function that is used to train using a single batch input into the
    autoencoder. A partial training loss can be calculated using this method
    """
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values,  this is
    # unsupervised learning)
    for image_batch, _ in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % loss.data)
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def train_batchs(encoder, decoder, device, dataloader, loss_fn, optimizer):
    """
    Function that is used to train the Convolutional Autoencoder and return
    the mean loss, averaged over all input batches.
    """
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    # Iterate the dataloader (we do not need the label values, this is
    # unsupervised learning)
    for image_batch, _ in dataloader:
        image_batch = image_batch.unsqueeze(0).permute(1, 0, 2, 3)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    # Train Accuracy
    return np.mean(train_loss)


def test_batchs(encoder, decoder, device, dataloader, loss_fn):
    """
    Function that is used to test the Convolutional Autoencoder and return the
    mean loss, averaged over all input batches.
    """
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        output_list = []
        label_list = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.unsqueeze(0).permute(1, 0, 2, 3)
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            output_list.append(decoded_data.cpu())
            label_list.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        output_list = torch.cat(output_list)
        label_list = torch.cat(label_list)
        # Evaluate global loss
        val_loss = loss_fn(output_list, label_list)
    return val_loss.data


def plot_autoencoder_results(encoder, decoder, dataset, n=5):
    """
    Function that plots the original and reconstructed images form the
    autoencoder results
    """
    torch.manual_seed(0)
    plt.figure(figsize=(10, 4.5))

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = dataset.train_dataset[i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')

    plt.show()


def create_encoded_samples_dataframe(folder_path, transform=None,
                                     latent_dims=120, save=False,
                                     encoded_samples_file_path=None):
    """
    Function that  uses the autoencoder to encode the samples and return an
    encoded samples dataframe to the user based on latent dimensions input by
    the user
    """
    encoded_samples_list = []
    dataset = Image_Loader.generate_dataset(folder_path=folder_path,
                                            transform=transform)
    for sample in tqdm(dataset.dataset):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        encoder, decoder = Image_Loader.initialize_encoder_decoder(
            latent_dimensions=latent_dims)
        encoder.eval()
        with torch.no_grad():
            encoded_img = encoder(img)
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in
                          enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples_list.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples_list)
    len_samples = len(encoded_samples)
    if save:
        if encoded_samples_file_path[-1] == "/":
            encoded_samples.to_csv(
                encoded_samples_file_path + "encoded_samples_" + str(
                    len_samples) + "_" + str(latent_dims) + ".csv")
        else:
            encoded_samples.to_csv(
                encoded_samples_file_path + "/" + "encoded_samples_" + str(
                    len_samples) + "_" + str(latent_dims) + ".csv")

    return encoded_samples


"""
    Transfer Learning functions
"""


def transfer_learning_encoding(folder_path, encoded_samples_file_path=None,
                               transfer_model="InceptionV3", test=False,
                               verbose=False, save=False):
    """
    The Transfer learning function takes in the folder path of the images to
     be encoded and uses either
    inceptionV3 or Xception, as specificed by the user to return an encoded
    features dataframe of the image samples
    """
    # Define the standard transforms for the images
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256)),
         transforms.Normalize(0.40655, 0.1159),
         transforms.Grayscale(num_output_channels=3)])
    # Initialize the Image Client
    dataset_all = Image_Loader.generate_dataset(folder_path=folder_path,
                                                transform=transform)
    image_list = utils.create_image_list(file_path=folder_path,
                                         transform=transform)

    # Select appropriate base model depending on the user's choice
    if transfer_model == "InceptionV3":
        base_model = InceptionV3(input_shape=(256, 256, 3), weights='imagenet',
                                 include_top=False)
    if transfer_model == "Xception":
        base_model = Xception(input_shape=(256, 256, 3), weights='imagenet',
                              include_top=False)

    # Freeze layers to be left untrained
    for layer in base_model.layers:
        layer.trainable = False

    # Import the weights and the pretrained model
    pretrained_model = Model(inputs=base_model.inputs,
                             outputs=base_model.layers[-2].output)

    if verbose:
        print(pretrained_model.summary())

    # Extract and append the features encoded using the pretrained model
    feature_list = []
    for img in image_list:
        image = img_to_array(img)

        image = preprocess_input(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        features = pretrained_model.predict(image)
        feature_list.append(features)

    # Create the output dataframe for the encoded features
    len_feature = len(feature_list[0].reshape(-1))
    cols = [f"Feature Var {i}" for i in range(len_feature)]
    feature_df = pd.DataFrame(columns=cols)

    for i in range(len(feature_list)):
        feature = feature_list[i].reshape(-1)
        feature_df.loc[len(feature_df)] = feature

    len_samples = len(feature_list)

    if test:
        # Create the output dataframe for the labels list
        label_list = []
        labels_df = pd.DataFrame(columns=['label'])

        for sample in tqdm(dataset_all.dataset):
            label = sample[1]
            label_list.append(label)
        labels_df['label'] = label_list

        if save:
            # Save the Dataframe and the label list in the appropriate format
            if encoded_samples_file_path[-1] == "/":
                feature_df.to_csv(
                    encoded_samples_file_path + "encoded_samples_" + str(
                        len_samples) + "_" + transfer_model + ".csv")
                labels_df.to_csv(
                    encoded_samples_file_path + "label_list_" + str(
                        len_samples) + "_" + transfer_model + ".csv")
            else:
                feature_df.to_csv(
                    encoded_samples_file_path + "/" + "encoded_samples_" + str(
                        len_samples) + "_" + transfer_model + ".csv")
                labels_df.to_csv(
                    encoded_samples_file_path + "/" + "label_list_" + str(
                        len_samples) + "_" + transfer_model + ".csv")
        return feature_df, labels_df

    else:
        if save:
            # Save the Dataframe and the label list in the appropriate format
            if encoded_samples_file_path[-1] == "/":
                feature_df.to_csv(
                    encoded_samples_file_path + "encoded_samples_" + str(
                        len_samples) + "_" + transfer_model + ".csv")
            else:
                feature_df.to_csv(
                    encoded_samples_file_path + "/" + "encoded_samples_" + str(
                        len_samples) + "_" + transfer_model + ".csv")
        return feature_df
