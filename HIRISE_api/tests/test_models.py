import numpy as np
import pkg_resources
from models import (
    KMeans,
    BIRCH,
    Agglomerative_Clustering,
    metrics,
    utils as mUtils
)
from preprocessing import utils
from models import (
    Affinity_Propagation,
    DBSCAN,
    HDBSCAN,
    OPTICS,
    Mean_Shift
)
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
    agg_clustering = (
        Agglomerative_Clustering.agglomerative_clustering_analysis(
            encoded_samples, clusters=14, plot=False, fig_size=(10, 10)
        )
    )
    affinity_clustering = \
        Affinity_Propagation.affinity_propagation_analysis(
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
    mean_shift_clustering = Mean_Shift.mean_shift_analysis(
        encoded_samples, plot=True
    )
    optics_clustering = OPTICS.OPTICS_analysis(
        dataframe=encoded_samples,
        eps=0.5,
        min_samples=5,
        plot=False,
        verbose=False,
    )

    # Assert if valid outputs are produced of finite length
    assert KMeans_clustering.all() is not None
    assert birch_clustering.all() is not None
    assert agg_clustering.all() is not None
    assert affinity_clustering.all() is not None
    assert dbscan_clustering.all() is not None
    assert hdbscan_clustering.all() is not None
    assert mean_shift_clustering.all() is not None
    assert optics_clustering.all() is not None


def test_translate_labels():
    """
    Test if label translation function for classification metrics translated
    user defined labels as expected
    """
    # Define random translation values
    model_results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    translation_list = [8, 7, 5, 11, 9, 2, 10, 3, 4, 1, 6, 13, 12]

    # Calculated translated dataframe
    translated_outputs = mUtils.translate_labels(
        translation_list, model_results
    )

    assert (
        translated_outputs == [9, 5, 7, 8, 2, 10, 1, 0, 4, 6, 3, 12, 11]
    ).all()


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
    test_length = len(
        metrics.calculate_metrics(
            model=model_results, labels=labels, verbose=False
        )
    )

    assert (
        test_length == 9
    )
