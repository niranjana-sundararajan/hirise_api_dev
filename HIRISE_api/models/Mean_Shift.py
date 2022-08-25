from numpy import unique
from numpy import where
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

def mean_shift_analysis(encoded_samples, plot = True, verbose = False, fig_size = (10,7)):
  X = StandardScaler().fit_transform(encoded_samples)

  # define the model
  mean_model = MeanShift()

  # assign each data point to a cluster
  mean_result = mean_model.fit_predict(encoded_samples)

  # get all of the unique clusters
  mean_clusters = unique(mean_result)
  if verbose:
    print("Number of clusters are :", len(mean_clusters))
  if plot:
    plt.figure(figsize= fig_size)
    clus_labels = [f"Clus. {i}" for i in mean_clusters]
    # plot Mean-Shift the clusters
  for mean_cluster in mean_clusters:
      # get data points that fall in this cluster
      index = where(mean_result == mean_cluster)
      # make the plot
      len_colors = len(X)
      colours = mean_result[0:len_colors]
      plt.scatter(X[index,0], X[index,1])
      plt.title("Mean Shift Analysis")
      plt.xlabel("Component 1")
      plt.ylabel("Component 2")
      plt.legend(clus_labels)

    # show the Mean-Shift plot
  plt.show()
  return mean_result