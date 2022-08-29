

from models import KMeans,BIRCH,Agglomerative_Clustering
from preprocessing import utils
from models import Affinity_Propagation,DBSCAN,HDBSCAN,OPTICS,Mean_Shift,Ensemble_Models
import pandas as pd

ENCODED_SAMPLES_CSV = 'encoded_samples.csv'
LABELS_CSV = 'label_list.csv'


def test_clustering_results():
    encoded_samples = utils.read_encoded_csv(ENCODED_SAMPLES_CSV)
    labels = utils.read_encoded_csv(LABELS_CSV)

    KMeans_clustering = KMeans.kmeans_analysis(encoded_samples, clusters = 14, plot = False, plot_centers = False)

    birch_clustering =BIRCH.BIRCH_analysis(encoded_samples, threshold_value = 0.2, clusters = 14, plot = False)

    agg_clustering = Agglomerative_Clustering.agglomerative_clustering_analysis(encoded_samples, clusters = 14, plot = False, fig_size = (10,10))

    affinity_clustering = Affinity_Propagation.affinity_propagation_analysis(encoded_samples, damping = 0.7, plot = False, fig_size= (15,10))

    dbscan_clustering = DBSCAN.DBSCAN_analysis(encoded_samples, true_labels=labels.label,eps = 0.5, min_samples = 9, verbose = False, plot = False)

    hdbscan_clustering = HDBSCAN.HDBSCAN_analysis(encoded_samples,  minimum_samples = 5, verbose = False, plot = False)

    mean_shift_clustering= Mean_Shift.mean_shift_analysis(encoded_samples, plot = True)

    optics_clustering = OPTICS.OPTICS_analysis(dataframe = encoded_samples,eps=0.5 ,min_samples=5, plot = False, verbose = False )

    
    assert isinstance(KMeans_clustering, list)
    assert len(KMeans_clustering)!=0
    assert isinstance(birch_clustering, list)
    assert len(birch_clustering)!=0
    assert isinstance(agg_clustering, list)
    assert len(agg_clustering)!=0
    assert isinstance(affinity_clustering, list)
    assert len(affinity_clustering)!=0
    assert isinstance(dbscan_clustering, list)
    assert len(dbscan_clustering)!=0
    assert isinstance(hdbscan_clustering, list)
    assert len(hdbscan_clustering)!=0
    assert isinstance(mean_shift_clustering, list)
    assert len(mean_shift_clustering)!=0
    assert isinstance(optics_clustering, list)
    assert len(optics_clustering)!=0