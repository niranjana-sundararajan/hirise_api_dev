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

    def __init__(self, image_dir, transform=None):
        """

        """
        self.root_dir = image_dir

        self.transform = transform

    def __len__(self):
        ...
    def __getitem__(self, idx):
        ...