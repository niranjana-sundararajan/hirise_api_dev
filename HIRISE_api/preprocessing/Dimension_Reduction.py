from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import umap.umap_ as umap

os.environ['OPEgrid_columnsV_IO_ENABLE_JASPER'] = 'true'

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


def PCA_analysis(encoded_samples, labels, components=30, explained_variance=False, cum_explained_variance=True,
                 plot_2d=False, plot_3d=False, fig_size=(5, 5)):
    if cum_explained_variance:
        pca = PCA(n_components=components)
        plt.figure(figsize=fig_size)
        plt.title("Cumulative Explained Variance")
        plt.xlabel('Number of Components')
        plt.ylabel('Percent Explained')
        plt.plot(range(1, components + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        print(f"The Cumulative explained variance with {components} is given by: ",
              pca.explained_variance_ratio_.cumsum()[-1] * 100, "%")

    pca = PCA(n_components=components)
    principal_components = pca.fit_transform(encoded_samples)
    if explained_variance:
        print(f"The Explained variance by the {components} components are :", pca.explained_variance_ratio_)
    cols = ["pca" + str(i) for i in range(1, components + 1)]
    pca_df = pd.DataFrame(data=principal_components, columns=cols)

    if plot_2d:
        plt.figure(figsize=fig_size)
        plt.title("Principal Component Analysis")
        plt.xlabel('pca1')
        plt.ylabel('pca2')
        ax = sns.scatterplot(x=pca_df['pca1'], y=pca_df['pca2'], hue=labels, data=principal_components,
                             palette='Accent')
        # plt.scatter( pca_df['pca1'], pca_df['pca2'],c =labels, cmap = 'Accent_r')
    if plot_3d:
        plt.figure(figsize=fig_size)
        ax = plt.axes(projection='3d')
        plt.title("Principal Component Analysis")
        plt.xlabel('pca1')
        plt.ylabel('pca2')
        plt.ylabel('pca3')
        ax.scatter(pca_df['pca3'], pca_df['pca1'], pca_df['pca2'], c=labels, cmap='Accent_r', linewidth=2)

    return pca_df


def TSNE_analysis(encoded_samples, labels, plot=True, plot_3d=False, fig_size=(12, 10)):
    if plot_3d:
        tsne = TSNE(n_components=3)
        tsne_results = tsne.fit_transform(encoded_samples)
        ax = plt.axes(projection='3d')
        plt.title("Latent Space After TSNE")
        ax.scatter(xs=tsne_results[:, 0], ys=tsne_results[:, 1], zs=tsne_results[:, 2], c=labels, cmap='Accent')
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples)
    if plot:
        plt.figure(figsize=fig_size)
        ax = sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, data=tsne_results,
                             palette='Accent')
        plt.title("Latent Space After TSNE")
        plt.legend(loc='upper right')
        plt.show()
    return tsne_results


def UMAP_analysis(encoded_samples, components=2, neighbours=10, training_epochs=1000, learning_rate=1, labels=None,
                  verbose=False, plot=False, plot_3d=False, fig_size=(10, 10)):
    # Configure UMAP hyperparameters
    reducer = umap.UMAP(n_neighbors=neighbours,
                        n_components=components,
                        metric='euclidean',
                        n_epochs=training_epochs,
                        learning_rate=learning_rate,
                        init='spectral',  # {'spectral', 'random', A numpy array of initial embedding positions}.
                        min_dist=0.1,  # minimum distance between embedded points.
                        spread=1.0
                        )

    # Fit and transform the data
    UMAP_results = reducer.fit_transform(encoded_samples)
    cols = ["UMAP" + str(i) for i in range(1, components + 1)]
    UMAP_df = pd.DataFrame(data=UMAP_results, columns=cols)

    if plot:
        plt.figure(figsize=fig_size)
        plt.title("Uniform Manifold Approximation and Projection")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.scatter(UMAP_df['UMAP1'], UMAP_df['UMAP2'], c=UMAP_df['UMAP2'], cmap='Accent_r')
    if plot_3d:
        plt.figure(figsize=fig_size)
        ax = plt.axes(projection='3d')
        plt.title("Uniform Manifold Approximation and Projection")
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.ylabel('UMAP3')
        ax.scatter(UMAP_df['UMAP1'], UMAP_df['UMAP2'], UMAP_df['UMAP3'], c=UMAP_df['UMAP2'], cmap='Accent_r',
                   linewidth=2)
    if verbose:
        print('Shape of X_trans: ', UMAP_results.shape)
    return UMAP_results
