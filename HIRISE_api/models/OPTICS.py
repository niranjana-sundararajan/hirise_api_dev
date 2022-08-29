from numpy import unique
from numpy import where
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def OPTICS_analysis(
    dataframe,
    eps=0.5,
    min_samples=15,
    plot=False,
    verbose=False,
    fig_size=(10, 7),
):
    """
    Function that uses as input the encoded image samples and clusters the
    data using  Ordering Points To Identify Cluster Structure Method.
    The user must specify the eps and minimum number of samples which are
    the  tuning parameter for OPTICS.
    """
    # Standardize the encoded samples
    X = StandardScaler().fit_transform(dataframe)

    # Instantiate the model
    optics_model = OPTICS(eps=eps, min_samples=min_samples)
    # assign each data point to a cluster
    optics_result = optics_model.fit_predict(dataframe)

    # get all the unique clusters
    optics_clusters = unique(optics_result)
    if verbose:
        print("Number of Clusters : ", len(optics_clusters))

    if plot:
        plt.figure(figsize=fig_size)
        cols = [f"Clus {optics_cluster}" for optics_cluster in optics_clusters]
        # plot OPTICS the clusters
        for optics_cluster in optics_clusters:
            # get data points that fall in this cluster
            index = where(optics_result == optics_cluster)
            # make the plot
            plt.scatter(X[index, 0], X[index, 1])
            plt.title(
                "Ordering points to identify the clustering structure  ("
                "OPTICS) Analysis"
            )
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.legend(
                labels=cols,
                loc="upper right",
                bbox_to_anchor=(0.5, 0.0, 0.65, 0.5),
            )

        # Show OPTICS plot
        plt.show()
    return optics_result
