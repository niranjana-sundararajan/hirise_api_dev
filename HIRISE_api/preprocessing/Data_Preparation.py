from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
from glob import glob
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import cv2
import os
import math
import random
import numpy as np
import warnings  # Ignore warnings

os.environ["OPEgrid_columnsV_IO_ENABLE_JASPER"] = "true"
warnings.filterwarnings("ignore")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

if __package__ is None or __package__ == "":
    # uses current directory visibility
    import Image_Loader
else:
    from . import Image_Loader


class DataPreparation:
    """Class that allows for data preparation as part of the preprocessing of
    the hirise images."""

    def remove_background(self, file_name):
        src = cv2.imread(file_name, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(src)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        if np.allclose(np.asarray(dst), 0):
            os.remove(file_name)
        else:
            cv2.imwrite(file_name, dst)

    def resize_image(
            self, folder_path, resized_images_folder_path, pixel_length_cm=250
    ):
        # Reduction factor based on NASA's decription of the HIRISE image
        reduce_factor = 25 / pixel_length_cm
        imgfiles = glob(f"{folder_path}/*.IMG")

        # Convert to PIL Image
        img_list = []
        for img in tqdm(imgfiles):
            img_list.append(Image.open(img))

        # If folder path is present, change path to folder path, else create
        # folder path
        if os.path.isdir(resized_images_folder_path):
            os.chdir(resized_images_folder_path)
        else:
            os.makedirs(resized_images_folder_path)
            os.chdir(resized_images_folder_path)

        # For each image in folder, resize image
        for im, name in tqdm(zip(img_list, imgfiles)):
            resized_im = im.resize(
                (
                    round(im.size[0] * reduce_factor),
                    round(im.size[1] * reduce_factor),
                )
            )
            try:
                resized_im.save(name.split("/")[-1] + "_resizedimage.jpg")
            except (Exception,):
                pass

    def tile_images(
            self,
            folder_path,
            image_directory,
            image_size_pixels,
            resized=True,
            remove_background=True,
    ):
        """
        Preprocessing function that tiles large images based on the size
        specified by the user
        """

        # Check if the files are resizedor in the original format
        if resized:
            imgfiles = glob(f"{folder_path}/*.jpg")
        else:
            imgfiles = glob(f"{folder_path}/*.IMG")

        # Convert to PIL Image
        img_list = []
        for img in imgfiles:
            img_list.append(Image.open(img))

        # If folder path is present, change path to folder path, else create
        # folder path
        if os.path.isdir(image_directory):
            os.chdir(image_directory)
        else:
            os.makedirs(image_directory)
            os.chdir(image_directory)

        # For each image, check size and tile accordingly
        for img, name in tqdm(zip(img_list, imgfiles)):
            try:
                im = np.asarray(img)
                for r in range(0, math.ceil(im.shape[0]), image_size_pixels):
                    for c in range(
                            0, math.ceil(im.shape[1]), image_size_pixels
                    ):
                        f_name = (
                                name.split("/")[-1].split(".")[
                                    0] + f"_{r}_{c}.jpg"
                        )
                        cv2.imwrite(
                            str(f_name),
                            im[
                                r: r + image_size_pixels,
                                c: c + image_size_pixels,
                                :,
                            ],
                        )
                        if remove_background:
                            DataPreparation.remove_background(
                                self, file_name=f_name
                            )
            except (Exception,):
                pass

    def convert_to_grayscale(
            self, folder_path, image_directory, remove_background=True
    ):
        """
        Preprocessing function that converts images into grayscale images
        """
        imgfiles = glob(f"{folder_path}/*.jpg")

        im_list = []
        # Convert to PIL Image
        for img in imgfiles:
            im_list.append(cv2.imread(img, 1))

        if os.path.isdir(image_directory):
            os.chdir(image_directory)
        else:
            os.makedirs(image_directory)
            os.chdir(image_directory)

        for img, name in zip(im_list, imgfiles):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f_name = "gray_" + name.split("\\")[1]
            try:
                cv2.imwrite(f_name, gray)
            except (Exception,):
                pass
            if remove_background:
                DataPreparation.remove_background(self, file_name=f_name)

    def remove_image_with_empty_pixels(
            self, folder_path, max_percentage_empty_space=20
    ):
        """
        Preprocessing function that removes tiled images with a specified
        percentage of empty pixels, to avoid noise in the dataset
        """
        # Get file names
        imgfiles = glob(f"{folder_path}/*.jpg")

        # If folder not present, throw error
        if os.path.isdir(folder_path):
            os.chdir(folder_path)
        else:
            print("ERROR!")

        # Extract and check each image in the folder for their % empty space
        for f_name in tqdm(imgfiles):
            empty = 0
            try:
                img = Image.open(f_name.split("\\")[-1])
            except IOError:
                img = Image.open(f_name.split("/")[-1])

            width, height = img.width, img.height
            total = width * height
            for pixel in img.getdata():
                if pixel == (0, 0, 0, 0) or pixel == (0, 0, 0):
                    empty += 1
            percent = round((empty * 100.0 / total), 1)

            # Remove images that have empty sapce above specified amount
            if percent >= max_percentage_empty_space:
                try:
                    os.remove(f_name.split("\\")[1])
                except IOError:
                    os.remove(f_name)

    def get_image_dataset(self, f_path, transform_data=None):
        """
        Function that returns the pytorch Image Loader dataset for images in a
        specified folder path
        """
        if not transform_data:
            transform_data = transforms.Compose([transforms.ToTensor()])
        dataset = Image_Loader.HiriseImageDataset(
            path_to_images=f_path, transform=transform_data
        )

        return dataset

    def get_train_test_val_tensors(self, dataset):
        """
        Function that returns the pytorch tensors for train, test and
        validation data with dataset a Imageloader dataset as the input
        """

        # Split the dataset into training and validation
        m = len(dataset.train_dataset)
        train_ds, val_ds = random_split(
            dataset.train_dataset,
            [math.floor(m - m * 0.2), math.ceil(m * 0.2)],
        )

        # Training Data
        # Empty lists to store the training data
        train_list = []
        # Append from the MedicalMNIST Object the training target and labels
        for data in train_ds:
            train_list.append(data[0])

        train_tensor = torch.Tensor(len(train_list))
        try:
            torch.cat(train_list, out=train_tensor)
        except (Exception,):
            pass

        # Test Data ------------------------
        # Empty lists to store the test data
        test_list = []
        for data in dataset.test_dataset:
            test_list.append(data[0])

        test_tensor = torch.Tensor(len(test_list))
        try:
            torch.cat(test_list, out=test_tensor)
        except (Exception,):
            pass

        # Validaton data ------------------
        # Empty lists to store the test data
        val_list = []
        for data in val_ds:
            val_list.append(data[0])

        val_tensor = torch.Tensor(len(val_list))

        try:
            torch.cat(val_list, out=val_tensor)
        except (Exception,):
            pass
        return train_tensor, test_tensor, val_tensor

    def get_dataset_tensor(self, dataset):
        tensor_list = []

        for data in dataset.dataset:
            tensor_list.append(data[0])

        dataset_tensor = torch.Tensor(len(tensor_list))
        try:
            torch.cat(tensor_list, out=dataset_tensor)
        except (Exception,):
            pass

        return dataset_tensor

    def get_train_test_val_dataloader(
            self, train_data, test_data, val_data, b_size=128
    ):
        """
        Function that returns the pytorch dataloader for train, test and
        validation data with batch size as the input parameter
        """
        # Create TorchTensor Datasets containing training_data, testing_data,
        # validation_data
        training_data = TensorDataset(train_data, train_data.long())
        validation_data = TensorDataset(val_data, val_data.long())
        testing_data = TensorDataset(test_data, test_data.long())
        # Create dataloaders
        train_loader = DataLoader(dataset=training_data, batch_size=b_size)
        valid_loader = DataLoader(dataset=validation_data, batch_size=b_size)
        test_loader = DataLoader(
            dataset=testing_data, batch_size=b_size, shuffle=True
        )

        return train_loader, test_loader, valid_loader

    def show_training_data(self, dataset, grid_rows=5, grid_columns=5):
        """
        Prints the training images in a defined grid
        """
        # Set up axes and subplots
        fig, axarr = plt.subplots(grid_rows, grid_columns, figsize=(10, 10))

        # Loops to run over the grid
        for i in range(grid_rows):
            for j in range(grid_columns):

                # Generate a random index in the training dataset
                idx = random.randint(0, len(dataset.train_dataset))

                # Get the sample and target from the training datasets
                sample, target = dataset.train_dataset[idx]

                try:
                    # Exception handling - if it is PIL
                    axarr[i][j].imshow(sample, cmap="gray")
                except (Exception,):
                    # If tensor of shape CHW
                    axarr[i][j].imshow(sample.permute(1, 2, 0), cmap="gray")
                    # Get the classes of the target data
                target_name = dataset.dataset.targets[target]
                # Label each image with the target name and the class it
                # belongs to
                axarr[i][j].set_title("%s (%i)" % (target_name, target))
        # Define the grid layout and padding

        fig.tight_layout(pad=1)
        plt.show()
