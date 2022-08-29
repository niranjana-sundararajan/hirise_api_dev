from numpy import unique
from numpy import where
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def BIRCH_analysis(
    encoded_samples,
    threshold_value,
    clusters,
    plot=False,
    fig_size=(10, 10),
    verbose=False,
):
    """
    Function that uses as input the encoded image samples and clusters the
    data using affinity propogation.
    The user must specify the damping factor, which is the tuning parameter
    for affinity propogation clustering.
    """
    # Standardize the encoded samples
    X = StandardScaler().fit_transform(encoded_samples)

    # Instantiate the model
    birch_model = Birch(threshold=threshold_value, n_clusters=clusters)

    # Train the model
    birch_model.fit(encoded_samples)

    # Assign each data point to a cluster
    birch_result = birch_model.predict(encoded_samples)

    # Get all the unique clusters
    birch_clusters = unique(birch_result)

    # Display additional information, if specified
    if verbose:
        print("Number of Clusters : ", len(birch_clusters))

    # Plotting, if specified
    if plot:
        labels = [f"Cluster {i}" for i in range(len(birch_result))]
        plt.figure(figsize=fig_size)

        # plot the BIRCH clusters
        for birch_cluster in birch_clusters:
            # get data points that fall in this cluster
            index = where(birch_result == birch_cluster)

            # make the plot
            plt.scatter(X[index, 0], X[index, 1])
            plt.legend(labels, loc="upper right")

        # show the BIRCH plot
        plt.show()

    return birch_result
