from preprocessing.Encoding import CAEEncoder
from torchsummary import summary

import torch


def test_autoencoder():
    # Dummy input of the same size as the HIRISE images
    x = torch.randn(1,1,256,256) 

    # Instantiate the model
    model = CAEEncoder.CAE_Encoder(encoded_space_dim=8192,fc2_input_dim=256)
    # Run the model
    y = model(x) 
    
    x_shape = x.shape
    y_shape =  y.shape
    print(x_shape[2], x_shape[3])
    print(y_shape[2], y_shape[3])



test_autoencoder()