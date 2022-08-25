from numpy import unique
from numpy import where
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

def affinity_propagation_analysis(encoded_samples,damping, plot = True,fig_size = (7,7) ):
  X = StandardScaler().fit_transform(encoded_samples)
  # define the model
  model = AffinityPropagation(damping=damping)

  # train the model
  model.fit(encoded_samples)

  # assign each data point to a cluster
  affinity_result = model.predict(encoded_samples)

  # get all of the unique clusters
  affinity_clusters = unique(affinity_result)
  print("Number of Clusters :", len(affinity_clusters))
  if plot :
    plt.figure(figsize= fig_size)
    labels = [f"Clus {i}" for i in range(len(affinity_result))]
    # plot the clusters
    for cluster in affinity_clusters:
        # get data points that fall in this cluster
        index = where(affinity_result == cluster)
        # make the plot
        pyplot.scatter(X[index,0], X[index,1])
        plt.legend(labels, loc= 'center left', bbox_to_anchor=(1, 0.5))
        plt.title("Affinity Propagation on Extracted Features")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    # show the plot
    plt.show()

  return affinity_result