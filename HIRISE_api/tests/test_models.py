import black
import numpy as np

import models
from models import KMeans, BIRCH, Agglomerative_Clustering, metrics
from preprocessing import utils
from models import (
    Affinity_Propagation,
    DBSCAN,
    HDBSCAN,
    OPTICS,
    Mean_Shift,
    Ensemble_Models,
)
import pandas as pd

ENCODED_SAMPLES_CSV = "./encoded_samples.csv"
LABELS_CSV = "./label_list.csv"


def test_clustering_results():

    # Read encoded samples and labels
    encoded_samples = utils.read_encoded_csv(ENCODED_SAMPLES_CSV)
    labels = utils.read_encoded_csv(LABELS_CSV)

    # Check if each model functions as expected
    KMeans_clustering = KMeans.kmeans_analysis(
        encoded_samples, clusters=14, plot=False, plot_centers=False
    )
    birch_clustering = BIRCH.BIRCH_analysis(
        encoded_samples, threshold_value=0.2, clusters=14, plot=False
    )
    agg_clustering = Agglomerative_Clustering.agglomerative_clustering_analysis(
        encoded_samples, clusters=14, plot=False, fig_size=(10, 10)
    )
    affinity_clustering = Affinity_Propagation.affinity_propagation_analysis(
        encoded_samples, damping=0.7, plot=False, fig_size=(15, 10)
    )
    dbscan_clustering = DBSCAN.DBSCAN_analysis(
        encoded_samples,
        true_labels=labels.label,
        eps=0.5,
        min_samples=9,
        verbose=False,
        plot=False,
    )
    hdbscan_clustering = HDBSCAN.HDBSCAN_analysis(
        encoded_samples, minimum_samples=5, verbose=False, plot=False
    )
    mean_shift_clustering = Mean_Shift.mean_shift_analysis(encoded_samples, plot=True)
    optics_clustering = OPTICS.OPTICS_analysis(
        dataframe=encoded_samples, eps=0.5, min_samples=5, plot=False, verbose=False
    )

    # Assert if valid outputs are produced of finite length
    assert KMeans_clustering
    assert birch_clustering
    assert agg_clustering
    assert affinity_clustering
    assert dbscan_clustering
    assert hdbscan_clustering
    assert mean_shift_clustering
    assert optics_clustering


def test_translate_labels():
    """
    Test if label translation function for classification metrics translated user defined labels as expected
    """
    # Define random translation values
    model_results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    translation_list = [8, 7, 5, 11, 9, 2, 10, 3, 4, 1, 6, 13, 12]

    # Calculated translated dataframe
    translated_outputs = models.utils.translate_labels(translation_list, model_results)

    assert (translated_outputs == [9, 5, 7, 8, 2, 10, 1, 0, 4, 6, 3, 12, 11]).all()


def test_metrics_function():
    """
    Test if metrics function returns all 9 metrics output as expected
    """

    # Generate random model results
    model_results = []
    for i in range(135):
        model_results.append(np.random.randint(0, 13))

    # Import labels file
    labels = utils.read_encoded_csv(LABELS_CSV)

    assert (
        len(
            metrics.calculate_metrics(model=model_results, labels=labels, verbose=False)
        )
        == 9
    )