from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def DBSCAN_analysis(encoded_samples, true_labels,eps = 0.5, min_samples = 5, verbose = True, plot = True, fig_size = (6,6)):
  X = StandardScaler().fit_transform(encoded_samples)
  DBSCAN_predictions = DBSCAN(eps = eps, min_samples = min_samples).fit_predict(encoded_samples)

  DBSCAN_clusters = np.unique(DBSCAN_predictions)

  if plot :
    clus_labels = {f"Clus. {i}" for i in DBSCAN_clusters}
    plt.figure(figsize=fig_size)
    ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue = DBSCAN_predictions, palette='viridis_r' )
    plt.title("Density-based clustering(DBSCAN)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(clus_labels)

  if verbose:
    print('Number of clusters: {}'.format(len(set(DBSCAN_predictions[np.where(DBSCAN_predictions != -1)]))))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(true_labels, DBSCAN_predictions)))
    print('Completeness: {}'.format(metrics.completeness_score(true_labels, DBSCAN_predictions)))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, DBSCAN_predictions))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(true_labels, DBSCAN_predictions))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(true_labels, DBSCAN_predictions))
  return DBSCAN_predictions