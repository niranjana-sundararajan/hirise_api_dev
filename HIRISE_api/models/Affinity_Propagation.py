from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

import matplotlib.pyplot as plt


def affinity_propagation_analysis(encoded_samples, damping, plot=True, fig_size=(7, 7), verbose=False):
    """
     Function that uses as input the encoded image samples and clusters the data using affinity propogation.
     The user must specify the damping factor, which is the tuning parameter for affinity propogation clustering.
     """
    X = StandardScaler().fit_transform(encoded_samples)
    # Define the model
    model = AffinityPropagation(damping=damping)

    # Train the model
    model.fit(encoded_samples)

    # Assign each data point to a cluster
    affinity_result = model.predict(encoded_samples)

    # Get all the unique clusters
    affinity_clusters = unique(affinity_result)

    if verbose:
        print("Number of Clusters :", len(affinity_clusters))

    if plot:
        plt.figure(figsize=fig_size)
        labels = [f"Clus {i}" for i in range(len(affinity_result))]
        # Plot the clusters
        for cluster in affinity_clusters:
            # Get data points that fall in this cluster
            index = where(affinity_result == cluster)

            pyplot.scatter(X[index, 0], X[index, 1])
            plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Affinity Propagation on Extracted Features")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")

        pyplot.show()

    return affinity_result
