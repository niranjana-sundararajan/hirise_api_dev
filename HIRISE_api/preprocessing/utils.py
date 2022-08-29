from PIL import Image

import numpy as np
import pandas as pd
import rasterfairy
import matplotlib.pyplot as plt
import math
import random
import torchvision.transforms as torchVisionTransforms

if __package__ is None or __package__ == "":
    import Dimension_Reduction
    import Image_Loader
else:
    from . import Dimension_Reduction
    from . import Image_Loader


def read_encoded_csv(file_path, autoencoder=False):
    encoded_samples_df = pd.read_csv(file_path)
    encoded_samples_df.drop("Unnamed: 0", axis=1, inplace=True)

    if autoencoder:
        labels_df = encoded_samples_df["label"]
        encoded_samples_df.drop("label", axis=1, inplace=True)
        return encoded_samples_df, labels_df
    else:
        return encoded_samples_df


def display_all_images(
    file_path,
    encoded_samples,
    labels,
    grid_rows=28,
    grid_columns=20,
    tile_width=72,
    tile_height=56,
    save=False,
    save_file_path=None,
    tsne=True,
    umap=False,
    pca=False,
):
    """
    Function to display all the image in the folder in a flat rasterfied
    format.
    The user may specify the dimension reduction used, default is t-SNE.
    The user must input the grid_rows, grid_columns, tile_width and
    tile_height  of the images and the samples to be diplayed.
    The grid_rows and coulumns must be specified such that
    total images <= grid_rows * grid_columns.
    The user may also choose to save the generated image in the current folder.
    """

    # Create an list of PIL images
    image_list = create_image_list(file_path=file_path)

    # Selecting correct dimenison redution method,
    if tsne:
        results = Dimension_Reduction.TSNE_analysis(
            encoded_samples=encoded_samples, labels=labels, plot=False
        )
    elif umap:
        results = Dimension_Reduction.UMAP_analysis(
            encoded_samples=encoded_samples, labels=labels, plot=False
        )
    elif pca:
        results = Dimension_Reduction.PCA_analysis(
            encoded_samples=encoded_samples, labels=labels
        )
    else:
        raise Exception("No analysis method specified")

    # Assign to grid
    grid_assignment = rasterfairy.transformPointCloud2D(
        results, target=(grid_rows, grid_columns)
    )

    # Calculate the full height and width based on given inputs
    full_width = tile_width * grid_rows
    full_height = tile_height * grid_columns

    # Calculate the aspect ratio
    aspect_ratio = float(tile_width) / tile_height

    # Format the image
    grid_image = Image.new("RGB", (full_width, full_height))

    # For each image, assign a position on the grid
    for img, grid_position in zip(image_list, grid_assignment[0]):
        index_x, index_y = grid_position
        x, y = tile_width * index_x, tile_height * index_y
        img_ar = float(img.width) / img.height
        if img_ar > aspect_ratio:
            margin = 0.5 * (img.width - aspect_ratio * img.height)
            img = img.crop(
                (margin, 0, margin + aspect_ratio * img.height, img.height)
            )
        else:
            margin = 0.5 * (img.height - float(img.width) / aspect_ratio)
            img = img.crop(
                (
                    0,
                    margin,
                    img.width,
                    margin + float(img.width) / aspect_ratio,
                )
            )
        img = img.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(img, (int(x), int(y)))

    plt.figure(figsize=(16, 12))
    plt.imshow(grid_image)
    if save:
        grid_image.save(save_file_path)


def normalize_results(encoded_samples):
    """
    Function that is used to normalize the values of the encoded samples.
    """
    tx, ty = encoded_samples[:, 0], encoded_samples[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    return tx, ty


def display_image_distributions(
    image_file_path, dim_red_results, width=3000, height=3000, max_dim=100
):
    """
    Function to display all the image in the folder as images on a
    distributed  map using TSNE,UMAP or PCA as the preprocessing function
    """
    # Create list of PIL images
    image_list = create_image_list(file_path=image_file_path)

    # Normalize the encoded imaegs
    tx, ty = normalize_results(encoded_samples=dim_red_results)
    full_image = Image.new("RGBA", (width, height))

    # For each image, assign a position on the grid
    for img, x, y in zip(image_list, tx, ty):
        tile = img
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize(
            (int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS
        )
        full_image.paste(
            tile,
            (int((width - max_dim) * x), int((height - max_dim) * y)),
            mask=tile.convert("RGBA"),
        )

    plt.figure(figsize=(16, 12))
    plt.title("Images After Dimension Reduction")
    plt.imshow(full_image)


def create_image_list(file_path, transform=None):
    """
    Function that creates a list of all the images in the specified folder
    in  a PIL format
    """
    # Generate the dataset
    full_dataset = Image_Loader.generate_dataset(
        folder_path=file_path, transform=transform
    )
    image_list = []

    # Define the PIL torch transform
    PIL_transform = torchVisionTransforms.ToPILImage()

    # Append the PIL images to the list
    for img, _ in full_dataset.dataset:
        im = PIL_transform(img)
        image_list.append(im)
    return image_list


def show_cluster_images(
    image_file_path,
    cluster_results,
    show_cluster_number,
    grid_rows=5,
    grid_columns=5,
    fig_size=(10, 10),
    all=False,
    test=False,
):
    """
    Prints the image in a specified cluster, in the form of a grid with rows
    and columns specified by the user
    """
    # Create list of PIL images
    image_list = create_image_list(file_path=image_file_path)

    # Define the cluster dataframe
    dataframe = pd.concat(
        [pd.Series(image_list), pd.DataFrame({"cluster": cluster_results})],
        axis=1,
    )

    # Filter and create the datafrmae for the specified cluster
    cluster_df = dataframe[dataframe["cluster"] == show_cluster_number]
    cluster_df = cluster_df.reset_index(drop=True)

    target_name = None
    hash_map = {}

    if all:
        grid_rows = math.ceil(len(cluster_df) / grid_columns)
        fig, axarr = plt.subplots(grid_rows, grid_columns, figsize=fig_size)
        idx = 0

        # Loops to run over the grid
        for i in range(grid_rows):
            for j in range(grid_columns):
                if idx < len(cluster_df):
                    img = cluster_df[0][idx]

                    try:
                        # Exception handling - if it is PIL
                        axarr[i][j].imshow(img, cmap="gray")
                    except (Exception,):
                        # If tensor of shape CHW
                        axarr[i][j].imshow(img, cmap="gray")

                    # Get the classes of the target data
                    target_name = cluster_df["cluster"][idx]

                    # Label each image with the target name and the class it
                    # belongs to
                    idx = idx + 1
                    axarr[i][j].set_xticks([])
                    axarr[i][j].set_yticks([])

    else:
        # Set up axes and subplots
        fig, axarr = plt.subplots(grid_rows, grid_columns, figsize=fig_size)

        # Loops to run over the grid
        for i in range(grid_rows):

            for j in range(grid_columns):
                # Generate a random index in the dataset
                idx = random.randint(0, len(cluster_df) - 1)

                if len(cluster_df) >= grid_rows * grid_columns:
                    while idx in hash_map and len(hash_map) <= len(cluster_df):
                        idx = random.randint(0, len(cluster_df) - 1)
                    hash_map[idx] = idx

                # Get the sample and target from the training dataset
                img = cluster_df[0][idx]

                try:
                    # Exception handling - if it is PIL
                    axarr[i][j].imshow(img, cmap="gray")
                except (Exception,):
                    # If tensor of shape CHW
                    axarr[i][j].imshow(img, cmap="gray")
                    # Get the classes of the target data
                target_name = cluster_df["cluster"][idx]

                # Label each image with the target name and the class it
                # belongs to
                axarr[i][j].set_xticks([])
                axarr[i][j].set_yticks([])

    # Define the grid layout and padding
    fig.tight_layout(pad=1)
    fig.subplots_adjust(top=0.95)

    # Add labels if the dataset is a test(labelled) dataset
    if test:
        dict_clust = Image_Loader.show_classes(
            folder_path=image_file_path, transform=None
        )
        res = dict((v, k) for k, v in dict_clust.items())
        fig.suptitle(
            " %s-images , total images in cluster = %s "
            % (res[target_name], len(cluster_df))
        )
    else:
        fig.suptitle(
            "Cluster number %s , images in cluster = %s "
            % (target_name, len(cluster_df))
        )

    plt.show()
