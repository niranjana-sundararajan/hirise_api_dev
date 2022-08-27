from torchvision import transforms
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.models import Model
from keras.utils import img_to_array
from keras.applications.inception_v3 import preprocess_input
from hirise.Image_Client import ImageClient
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

os.environ['OPEgrid_columnsV_IO_ENABLE_JASPER'] = 'true'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # Check if the GPU is available


class CAEEncoder(nn.Module):

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
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x


# Training function
def train_CAE(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
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


# Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
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


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.unsqueeze(0).permute(1, 0, 2, 3)
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs(encoder, decoder, dataset, n=5):
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


"""
    Transfer Learning functions
"""


def transfer_learning_encoding(self, folder_path="/content/drive/MyDrive/Images/test-data/",
                               transfer_model="InceptionV3", test=False, verbose=False):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256)), transforms.Normalize(0.40655, 0.1159),
         transforms.Grayscale(num_output_channels=3)])
    dataset_all = self.ImageClient.generate_dataset(folder_path=folder_path, transform=transform)
    image_list = self.ImageClient.create_image_list(file_path=folder_path, transform=transform)
    dp = self.ImageClient.Data_Preparation()
    dataset_tensor = dp.get_dataset_tensor(dataset_all)
    len_full_tensor = len(dataset_tensor.reshape(-1))
    dataset_tensor = dataset_tensor.reshape(int(len_full_tensor / (256 * 256)), 256 * 256)

    if transfer_model == "InceptionV3":
        base_model = InceptionV3(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    if transfer_model == "Xception":
        base_model = Xception(input_shape=(256, 256, 3), weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    pretrained_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    if verbose:
        print(pretrained_model.summary())

    feature_list = []
    for img in image_list:
        image = img_to_array(img)

        image = preprocess_input(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        features = pretrained_model.predict(image)
        feature_list.append(features)

    len_feature = len(feature_list[0].reshape(-1))
    cols = [f"Feature Var {i}" for i in range(len_feature)]
    feature_df = pd.DataFrame(columns=cols)

    for i in range(len(feature_list)):
        feature = feature_list[i].reshape(-1)
        feature_df.loc[len(feature_df)] = feature

    label_list = []
    labels_df = pd.DataFrame(columns=['label'])

    if test:
        for sample in tqdm(dataset_all.dataset):
            label = sample[1]
            label_list.append(label)
        labels_df['label'] = label_list
        return feature_df, labels_df
    else:
        return feature_df
