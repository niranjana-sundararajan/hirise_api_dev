import pandas as pd
import pkg_resources
from preprocessing import Encoding, Dimension_Reduction, utils
import numpy as np
import torch

if __package__ is None or __package__ == "":
    # uses current directory visibility
    ENCODED_SAMPLES_CSV = "./encoded_samples.csv"
    LABELS_CSV = "./label_list.csv"
else:
    ENCODED_SAMPLES_CSV = pkg_resources.resource_filename(
        "tests", "encoded_samples.csv"
    )
    LABELS_CSV = pkg_resources.resource_filename(
        "tests", "label_list.csv"
    )


def test_autoencoder():
    """
    Test if output size of the auto-encoded image is as expected
    """
    # Dummy input of the same size as the HIRISE images
    x = torch.randn(1, 1, 256, 256)

    # Instantiate the model
    model = Encoding.CAEEncoder(encoded_space_dim=8192, fc2_input_dim=256)

    # Run the model
    y = model(x)

    # Check the shape of the input and output
    x_shape = x.shape
    y_shape = y.shape

    # Assert if shapes are as expected
    assert x_shape[2] == 256
    assert y_shape[1] == 8192


def test_pca_dimension_reduction():
    """
    Test if output of PCA Analysis is as expected
    """
    encoded_samples = utils.read_encoded_csv(ENCODED_SAMPLES_CSV)
    labels = utils.read_encoded_csv(LABELS_CSV)
    pca_dimension_reduction = Dimension_Reduction.PCA_analysis(
        encoded_samples, labels, cum_explained_variance=False
    )

    assert isinstance(pca_dimension_reduction, pd.DataFrame)


def test_tsne_dimension_reduction():
    """
    Test if output of TSNE Analysis is as expected
    """
    encoded_samples = utils.read_encoded_csv(ENCODED_SAMPLES_CSV)
    labels = utils.read_encoded_csv(LABELS_CSV)
    tsne_dimension_reduction = Dimension_Reduction.TSNE_analysis(
        encoded_samples, labels, plot=False
    )

    assert isinstance(tsne_dimension_reduction, np.ndarray)


def test_umap_dimension_reduction():
    """
    Test if output of UMAP analysis is as expected
    """
    encoded_samples = utils.read_encoded_csv(ENCODED_SAMPLES_CSV)
    umap_dimension_reduction = Dimension_Reduction.UMAP_analysis(
        encoded_samples, plot=False
    )
    assert isinstance(umap_dimension_reduction, np.ndarray)
