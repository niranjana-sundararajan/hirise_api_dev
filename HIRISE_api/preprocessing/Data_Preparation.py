
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from glob import glob
import cv2
import os,sys
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
import numpy as np
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm
# Ignore warnings
import warnings
import math
warnings.filterwarnings("ignore")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from hirise import Image_Client
else:
    from hirise import Image_Client
# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

class Data_Preparation:
    """Class that allows for data prepartion as part of the preprocessing of the hirise images. """
    def remove_background(self,file_name):
        src = cv2.imread(file_name, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(src)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        if np.allclose(np.asarray(dst), 0):
            os.remove(file_name)
        else:
            cv2.imwrite(file_name, dst)

    def tile_images(self, folder_path,image_directory, image_size_pixels, remove_background = True):
        imgfiles = glob(f"{folder_path}/*.IMG")
        # Convert to PIL Imgae
        img_list = []
        for img in imgfiles:
            img_list.append(Image.open(img))

        if os.path.isdir(image_directory):
            os.chdir(image_directory)
        else:
            os.makedirs(image_directory)
            os.chdir(image_directory)
        img_count = 0
        for img in tqdm(img_list):
            img_count = img_count+1
            # im = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)
            im = np.asarray(img)
            for r in range(0,math.ceil(im.shape[0]),image_size_pixels):
                for c in range(0,math.ceil(im.shape[1]),image_size_pixels):

                        f_name = f"img_{img_count}_{r}_{c}.png"
                        cv2.imwrite(str(f_name), im[r:r+image_size_pixels, c:c+image_size_pixels,:] )
                        if remove_background:
                            Data_Preparation.remove_background(self,file_name =f_name)
                            

        sys.path.insert(0, parent_dir_path)

    def convert_to_grayscale(self, folder_path,image_directory, remove_background = True):
        imgfiles = glob(f"{folder_path}/*.png")

        im_list = []
        # Convert to PIL Imgae
        for img in imgfiles:
            im_list.append(cv2.imread(img, 1))
        
        if os.path.isdir(image_directory):
            os.chdir(image_directory)
        else:
            os.makedirs(image_directory)
            os.chdir(image_directory)

        for img,name in zip(im_list,imgfiles):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f_name = "gray_" + name.split('\\')[1]
            cv2.imwrite(f_name, gray)
            if remove_background:
                Data_Preparation.remove_background(self,file_name =f_name)
        sys.path.insert(0, parent_dir_path)

    def remove_image_with_empty_pixels(self, folder_path, max_percentage_empty_space = 50):
        imgfiles = glob(f"{folder_path}/*.png")

        if os.path.isdir(folder_path):
            os.chdir(folder_path)
        else:
            print("ERROR!!")

        for f_name in imgfiles:
            not_empty = 0
            img = Image.open(f_name.split('\\')[1])
            width, height = img.width, img.height
            total = width * height
            lower = 0
            higher = max_percentage_empty_space

            for pixel in img.getdata():
                if pixel != (0,0,0,0):
                    not_empty += 1
            percent = round((not_empty * 100.0/total),1)
            if((percent >= lower) & (percent < higher)):
                os.remove(f_name.split('\\')[1])
        sys.path.insert(0, parent_dir_path)

    def load_image_data():
        ...

    

dp = Data_Preparation()

# dp.tile_images(folder_path='./download-data',image_directory  ='./download-data/tiled-images/' , image_size_pixels = 256)

# dp.remove_background("img_37376_15616.png")
# dp.convert_to_grayscale(folder_path='./download-data/tiled-images',image_directory  ='./download-data/grayscale-images/')

dp.remove_image_with_empty_pixels(folder_path='./download-data/grayscale-images')