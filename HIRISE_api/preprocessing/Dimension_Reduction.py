import os,sys
from glob import glob
import cv2
os.environ['OPEgrid_columnsV_IO_ENABLE_JASPER'] = 'true'
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm
import random

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import torch packages that help us define our network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Normalize
from torchvision import datasets, transforms, models
import seaborn as sns
# Package that allows us to summarize our network
from torchsummary import summary

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


from sklearn.manifold import TSNE

def tsne_analysis(encoded_samples,labels, plot = True,plot_3d = False, fig_size = (12,10)):

  if plot_3d:
    tsne = TSNE(n_components=3)
    tsne_results = tsne.fit_transform(encoded_samples)
    ax = plt.axes(projection = '3d')
    plt.title("Latent Space After TSNE")
    ax.scatter(xs=tsne_results[:,0], ys=tsne_results[:,1],zs =tsne_results[:,2],  c = labels,  cmap='Accent' )
  tsne = TSNE(n_components=2)
  tsne_results = tsne.fit_transform(encoded_samples)
  if plot:
    plt.figure(figsize= fig_size)
    ax = sns.scatterplot( x=tsne_results[:,0], y=tsne_results[:,1], hue = labels, data = tsne_results, palette='Accent' )
    plt.title("Latent Space After TSNE")
    plt.legend(loc='upper right')
    plt.show()
  return tsne_results

def UMAP_analysis(encoded_samples,components = 2, neighbours = 10, training_epochs = 1000, learning_rate = 1, labels = None, verbose = False, plot = True, plot_3d = False,fig_size = (10,10)):

  # Configure UMAP hyperparameters
  reducer = umap.UMAP(n_neighbors=neighbours, 
                n_components= components, 
                metric='euclidean', 
                n_epochs=training_epochs,
                learning_rate=learning_rate, 
                init='spectral', # {'spectral', 'random', A numpy array of initial embedding positions}.
                min_dist=0.1, #  minimum distance between embedded points.
                spread=1.0 # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                )

  # Fit and transform the data
  UMAP_results = reducer.fit_transform(encoded_samples)
  cols = ["UMAP"+str(i) for i in range(1,components+1)]
  UMAP_df = pd.DataFrame(data = UMAP_results, columns = cols)

  if plot: 
      plt.figure(figsize = fig_size)
      plt.title("Uniform Manifold Approximation and Projection")
      plt.xlabel('Component 1')
      plt.ylabel('Componenet 2') 
      plt.scatter( UMAP_df['UMAP1'], UMAP_df['UMAP2'] ,c = UMAP_df['UMAP2'] ,cmap = 'Accent_r')
  if plot_3d:
      plt.figure(figsize = fig_size)
      ax = plt.axes(projection = '3d')
      plt.title("Uniform Manifold Approximation and Projection")
      plt.xlabel('UMAP1')
      plt.ylabel('UMAP2') 
      plt.ylabel('UMAP3')
      ax.scatter( UMAP_df['UMAP1'], UMAP_df['UMAP2'], UMAP_df['UMAP3'],c =UMAP_df['UMAP2'], cmap='Accent_r',linewidth = 2)
  if verbose:
    print('Shape of X_trans: ', UMAP_results.shape)
  return UMAP_results