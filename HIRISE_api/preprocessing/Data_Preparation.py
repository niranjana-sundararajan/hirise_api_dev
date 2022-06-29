
import pandas as pd
import torch
from torch.utils.data import TensorDataset,Dataset, DataLoader,random_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from glob import glob
import cv2
import os,sys
os.environ['OPEgrid_columnsV_IO_ENABLE_JASPER'] = 'true'
import numpy as np
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm
import random
# Ignore warnings
import warnings
import math
from Image_Loader import Hirise_Image_Dataset
warnings.filterwarnings("ignore")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUgrid_columnsATED_IMAGES = True

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
        for img,name in zip(img_list,imgfiles):
            try:
                im = np.asarray(img)
                for r in range(0,math.ceil(im.shape[0]),image_size_pixels):
                    for c in range(0,math.ceil(im.shape[1]),image_size_pixels):
                            f_name = name.split('\\')[1] + f"_{r}_{c}.png"
                            cv2.imwrite(str(f_name), im[r:r+image_size_pixels, c:c+image_size_pixels,:] )
                            if remove_background:
                                Data_Preparation.remove_background(self,file_name =f_name)
            except:
                pass                

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
            try:
                cv2.imwrite(f_name, gray)
            except:
                pass
            if remove_background:
                Data_Preparation.remove_background(self,file_name =f_name)
        sys.path.insert(0, parent_dir_path)

    def remove_image_with_empty_pixels(self, folder_path, max_percentage_empty_space = 20):
        imgfiles = glob(f"{folder_path}/*.png")

        if os.path.isdir(folder_path):
            os.chdir(folder_path)
        else:
            print("ERROR!!")

        for f_name in tqdm(imgfiles):
            empty = 0
            img = Image.open(f_name.split('\\')[1])
            width, height = img.width, img.height
            total = width * height
            for pixel in img.getdata():
                if pixel == (0,0,0,0):
                    empty += 1
            percent = round((empty * 100.0/total),1)
            if(percent >= max_percentage_empty_space):
                os.remove(f_name.split('\\')[1])
        sys.path.insert(0, parent_dir_path)

    def get_image_dataset(self,f_path,  transform_data =  None ):
        if not transform_data:
            transform_data = transforms.Compose([transforms.ToTensor()])
        # transform_data= transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
        dataset = Hirise_Image_Dataset(path_to_images = f_path, transform = transform_data)

        return dataset

    def get_train_test_val_tensors( dataset):
            m=len(dataset.train_dataset)

            train_ds, val_ds = random_split(dataset.train_dataset, [math.floor(m-m*0.2), math.ceil(m*0.2)])
            # ------------------Training Data ----------------------------------------------
            # Empty lists to store the training data
            train_list = []
            # Append from the MedicalMNIST Object the training target and labels
            for data in train_ds:
                train_list.append(data[0])

            train_tensor = torch.Tensor(len(train_list))
            try :
                torch.cat(train_list,out = train_tensor)
            except :
                pass
            # ------------------- --- Test Data ---------------------------------------------
            # Empty lists to store the test data
            test_list = []
            for data in dataset.test_dataset:
                test_list.append(data[0])

            test_tensor = torch.Tensor(len(test_list))
            try:
                torch.cat(test_list,out = test_tensor)
            except :
                pass
            # ------------------- --- Val Data ---------------------------------------------
            # Empty lists to store the test data
            val_list = []
            for data in val_ds:
                val_list.append(data[0])

            val_tensor = torch.Tensor(len(val_list))

            try:
                torch.cat(val_list,out = val_tensor)
            except :
                pass
            return  train_tensor, test_tensor, val_tensor


    def get_train_test_val_dataloader(self, train_data, test_data, val_data,  b_size = 128):
        # Create TorchTensor Datasets containing training_data, testing_data, validation_data
        training_data = TensorDataset(train_data)
        validation_data = TensorDataset(val_data)
        testing_data = TensorDataset(test_data)
        train_loader = DataLoader(dataset = training_data, batch_size=b_size)
        valid_loader = DataLoader(dataset = validation_data, batch_size=b_size)
        test_loader = DataLoader(dataset = testing_data, batch_size=b_size,shuffle=True)
        return train_loader,  test_loader, valid_loader

    def show_training_data(self, dataset, grid_rows=5, grid_columns=5):
        """ Prints the traning data in a grid"""
        # Set up axes and subplots
        fig, axarr = plt.subplots(grid_rows, grid_columns, figsize=(10, 10))

        # Loops to run over the grid
        for i in range(grid_rows):
            for j in range(grid_columns):

                # Generate a random index in the training dataset
                idx = random.randint(0, len(dataset.train_dataset))

                # Get the sample and target fromthe traiig dataset
                sample = dataset.train_dataset[idx]

                try:
                    # Exception handling - if it is PIL
                    axarr[i][j].imshow(sample, cmap = "gray")
                except:
                    # If tensor of shape CHW
                    axarr[i][j].imshow(sample.permute(1,2,0), cmap = "gray") 
                # # Get the classes of the target data
                # target_name = dataset.dataset.targets[target]
                # # Label each image eith the target name and the class it belongs to
                # axarr[i][j].set_title("%s (%i)"%(target_name, target))
        # Deine the grid layout and padding
        
        fig.tight_layout(pad=1)
        plt.show()



dp = Data_Preparation()
# dp.tile_images(folder_path='./download-data',image_directory  ='./download-data/tiled-images/' , image_size_pixels = 256)
# dp.remove_background("img_37376_15616.png")
# dp.convert_to_grayscale(folder_path='./download-data/tiled-images',image_directory  ='./download-data/grayscale-images/')
# dp.remove_image_with_empty_pixels(folder_path='./download-data/tiled-images/', max_percentage_empty_space = 10)

transform1= transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
dataset1 = dp.get_image_dataset(f_path ="./download-data/", transform_data = transform1)


# # print(dataset.test_dataset[10][0])

# print(dp.show_training_data(dataset= dataset1))
# print(len(dataset1.test_dataset))

tr,tst,val = Data_Preparation.get_train_test_val_tensors(dataset = dataset1)


tr_l,tst_l, val_l = Data_Preparation.get_train_test_val_dataloader(tr,tst,val )