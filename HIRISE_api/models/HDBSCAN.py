from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def HDBSCAN_analysis(encoded_samples, minimum_samples=15, plot=True, verbose=False, plot_tree=False, fig_size=(8, 5)):
    X = StandardScaler().fit_transform(encoded_samples)
    HDBSCAN_model = HDBSCAN(min_samples=minimum_samples, gen_min_span_tree=True)
    HDBSCAN_predictions = HDBSCAN_model.fit_predict(encoded_samples)
    number_of_clusters = np.unique(HDBSCAN_predictions) - 1
    if verbose:
        print(f"For {minimum_samples} min number of samples , the number of clusters found : ", len(number_of_clusters))
    if plot:
        clus_labels = {f"Clus. {i}" for i in number_of_clusters}
        plt.figure(figsize=fig_size)
        ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=HDBSCAN_predictions, palette='Accent')
        plt.title("Hierarchical Density Based Clustering (HDBSCAN)")
        plt.legend(labels=clus_labels)
    if plot_tree:
        HDBSCAN_model.minimum_spanning_tree_.plot()
        HDBSCAN_model.single_linkage_tree_.plot()
    return HDBSCAN_predictions
