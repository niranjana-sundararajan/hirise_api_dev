from numpy import unique
from numpy import where
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def agglomerative_clustering_analysis(
    encoded_samples, clusters, plot=False, fig_size=(10, 10)
):
    """
    Function that uses as input the encoded image samples and clusters the
    data using aglomerative clustering.
    The user must specify the number of clusters, which is a parameter for
    aglomerative clustering.
    """
    # Standardize the encoded samples
    X = StandardScaler().fit_transform(encoded_samples)

    # Instantiate the model
    agglomerative_model = AgglomerativeClustering(n_clusters=clusters)

    # Assign each data point to a cluster
    agglomerative_result = agglomerative_model.fit_predict(encoded_samples)

    # Get all the unique clusters
    agglomerative_clusters = unique(agglomerative_result)

    # Plotting, if specified
    if plot:
        labels = [f"Cluster {i}" for i in range(len(agglomerative_clusters))]
        plt.figure(figsize=fig_size)
        # plot the clusters
        for agglomerative_cluster in agglomerative_clusters:
            # get data points that fall in this cluster
            index = where(agglomerative_result == agglomerative_cluster)
            # make the plot
            plt.scatter(X[index, 0], X[index, 1])
            plt.legend(labels, loc="upper right")

        # show the Agglomerative Hierarchy plot
        plt.show()

    return agglomerative_result
