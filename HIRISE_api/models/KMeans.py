from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from numpy import where

import matplotlib.pyplot as plt


def kmeans_analysis(
    encoded_samples,
    clusters=13,
    plot_centers=False,
    fig_size=(5, 5),
    plot=True,
):
    """
    Function that uses as input the encoded image samples and clusters the
    data using K Means clustering of applications with noise.
    The user must specify the number of clusters and the algorithms chooses
    a random seed to instantiate the clustering.
    """
    # Standardize the encoded samples
    X = StandardScaler().fit_transform(encoded_samples)

    # Instantiate the model
    kmeans_encoded = KMeans(clusters, random_state=0)

    # Train the model
    kmeans_encoded.fit(encoded_samples)

    # Extract centroids
    centroids = kmeans_encoded.cluster_centers_

    # Train and predict the clusters
    kmeans_prediction_clusters = kmeans_encoded.fit_predict(encoded_samples)

    # Plotting centers,if specified
    if plot_centers:
        plt.figure(figsize=fig_size)
        ax = plt.axes(projection="3d")
        plt.title("K-Means Centroids")
        ax.scatter(
            centroids[:, 1],
            centroids[:, 2],
            centroids[:, 0],
            c=centroids[:, 0],
            cmap="Accent_r",
            linewidth=2,
        )

    # Plotting latent space,if specified
    if plot:
        plt.figure(figsize=fig_size)
        clus_labels = [f"Clus. {i}" for i in range(clusters)]
        # plot Mean-Shift the clusters
        for cluster in range(clusters):
            # get data points that fall in this cluster
            index = where(kmeans_prediction_clusters == cluster)
            plt.scatter(X[index, 0], X[index, 1])
            plt.legend(labels=clus_labels, loc="upper right")
        plt.title("K-Means Analysis")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

    return kmeans_prediction_clusters
