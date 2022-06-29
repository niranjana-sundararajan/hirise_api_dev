
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

if __package__ is None or __package__ == '':
    from preprocessing import Data_Preparation
else:
    from . import Data_Preparation

class CAE_Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class CAE_Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = CAE_Encoder()
        self.decoder = CAE_Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

class Hyperparameter:
    latent_dims = 10
    num_epochs = 50
    batch_size = 128
    capacity = 64
    learning_rate = 1e-3
    use_gpu = True

loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 4

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = CAE_Encoder(encoded_space_dim=d,fc2_input_dim=256)
decoder = CAE_Decoder(encoded_space_dim=d,fc2_input_dim=256)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


class Training:
    ### Training function
    def train_CAE(encoder, decoder, device, dataloader, loss_fn, optimizer):
        # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
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
            print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)
      ### Training function
    def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
                # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
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
            print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

class Validation:
    ...

class Testing:
    
    ### Testing function
    def test_epoch(encoder, decoder, device, dataloader, loss_fn):
        # Set evaluation mode for encoder and decoder
        encoder.eval()
        decoder.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch in dataloader:
                # Move tensor to the proper device
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
        return val_loss.dat
  


class Plot_losses:
    def plot_ae_outputs(encoder,decoder,n=10):
        plt.figure(figsize=(16,4.5))
        targets = Train_Model.tst.targets.numpy()
        t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            img = Train_Model.tst[t_idx[i]][0].unsqueeze(0).to(device)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                rec_img  = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Reconstructed images')
            plt.show()

class Latent_Space_Visualization:
    ...
def Train_Model(folder_path):
    num_epochs = 30
    diz_loss = {'train_loss':[],'val_loss':[]}
    transform1= transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    dp = Data_Preparation()
    dataset1 = dp.get_image_dataset(f_path = folder_path, transform_data = transform1)
    tr,tst,val = dp.get_train_test_val_tensors(dataset = dataset1)
    train_loader,test_loader, val_l = dp.get_train_test_val_dataloader(tr,tst,val )

    for epoch in range(num_epochs):
        train_loss =Training.train_epoch(encoder,decoder,device,
        train_loader,loss_fn,optim)
        val_loss = Testing.test_epoch(encoder,decoder,device,test_loader,loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        Plot_losses.plot_ae_outputs(encoder,decoder,n=10)



# Train_Model()