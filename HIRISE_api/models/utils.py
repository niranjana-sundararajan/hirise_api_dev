from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from preprocessing import Image_Loader

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def elbow_curve(encoded_samples, max_values=70, fig_size=(10, 5)):
    np.random.seed(0)
    inertia_values = []
    for i in tqdm(range(2, max_values)):
        kmeans_model = KMeans(n_clusters=i).fit(encoded_samples)
        inertia_values.append(kmeans_model.inertia_)

    plt.figure(figsize=fig_size)
    plt.plot(inertia_values, "bx-")
    plt.title("Finding the number of clusters: Elbow Method")
    plt.xlabel("Clusters")
    plt.ylabel("Scores")
    plt.show()


def translate_labels(translation_list, model_results):
    test_label_list = [i for i in range(14)]
    return np.array(
        [
            test_label_list[translation_list.index(i)]
            if i in translation_list
            else i
            for i in model_results
        ]
    )


def generate_precision_dataframe(
    folder_path, test_label_list, translated_model
):
    class_names = Image_Loader.show_classes(
        folder_path=folder_path, dict_values=False
    )
    precision_list = precision_score(
        test_label_list.label, translated_model, average=None
    )
    precision_df = pd.DataFrame(
        data=precision_list.reshape(1, 14), columns=class_names
    )
    return precision_df
