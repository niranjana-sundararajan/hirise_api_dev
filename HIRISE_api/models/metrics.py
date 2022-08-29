from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    mutual_info_score,
    rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
    balanced_accuracy_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

from preprocessing import Image_Loader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(model, labels, verbose=False):
    """
    Function that calulates metrics including, rand score, adjusted rand
    score, mutual information score, normalized mutual information score,
    adjusted mutual information score, balanced accuracy score, completeness
    score, homogeniety score and v-score for a given model. The label data
    must be provided by the user to
    obtain these metrics.
    """

    # Additional Information, if specified by the user.
    if verbose:
        print("RAND SCORE")
        print(rand_score(labels.label, model))
        print("ADJUSTED RAND SCORE")
        print(adjusted_rand_score(labels.label, model))
        print("MUTUAL INFO SCORE")
        print(mutual_info_score(labels.label, model))
        print("NORMALIZED MUTUAL INFO SCORE")
        print(normalized_mutual_info_score(labels.label, model))
        print("ADJUSTED MUTUAL INFO SCORE")
        print(adjusted_mutual_info_score(labels.label, model))
        print("BALANCED ACCURACY SCORE")
        print(balanced_accuracy_score(labels.label, model))
        print("COMPLETENESS SCORE")
        print(completeness_score(labels.label, model))
        print("HOMOGENIETY SCORE")
        print(homogeneity_score(labels.label, model))
        print("V SCORE")
        print(v_measure_score(labels.label, model))

    # Calcultion of metrics
    rs = rand_score(labels.label, model)
    ars = adjusted_rand_score(labels.label, model)
    mis = mutual_info_score(labels.label, model)
    nmis = normalized_mutual_info_score(labels.label, model)
    amis = adjusted_mutual_info_score(labels.label, model)
    bas = balanced_accuracy_score(labels.label, model)
    cs = completeness_score(labels.label, model)
    hs = homogeneity_score(labels.label, model)
    vs = v_measure_score(labels.label, model)
    return [rs, ars, mis, nmis, amis, bas, cs, hs, vs]


def classification_metrics_dataframe(model_list, label_list):
    """
    Fucntion that creates a metrics dataframe based on the calculated
    metrics for each model in the model list specified by the user.
    """
    metrics_list = []

    # Calcualte scores for each model in the list
    for mod in model_list:
        metrics_list.append(calculate_metrics(model=mod, labels=label_list))

    # Define the metrics dataframe
    metrics_dataframe = pd.DataFrame(
        metrics_list,
        columns=[
            "RAND_SCORE",
            "ADJUSTED RAND SCORE",
            "MUTUAL INFO SCORE",
            "NORMALIZED MUTUAL INFO SCORE",
            "ADJUSTED MUTUAL INFO SCORE",
            "BALANCED ACCURACY SCORE",
            "COMPLETENESS SCORE",
            "HOMOGENEITY SCORE",
            "V SCORE",
        ],
    )
    return metrics_dataframe


def print_confusion_matrix(
    folder_path, test_labels, translated_values_array, fig_size=(15, 10)
):
    """
    Function that prints the confusion matrix metric for a given set of
    image  clustering results and the associated images.
    The translated vlaue aray must be entered by the user which represents
    the  results of manual inspection of the clusters
    by the user based on corresonding defined class number/name pairs in the
    test-dataset
    """
    # Generate Image Loader Dataset
    dataset1 = Image_Loader.generate_dataset(folder_path)

    # Generate list of class names and indices
    class_names = dataset1.dataset.class_to_idx

    # Generate confusion matrix for the correctly translated pairs of true and
    # predicted labels
    cm = confusion_matrix(test_labels.label, translated_values_array)

    # Generate a dataframe for the confusion matrix
    df_cm = pd.DataFrame(cm)

    # Plot figures using matplot
    plt.figure(figsize=fig_size)
    sns.heatmap(
        df_cm, annot=True, xticklabels=class_names, yticklabels=class_names
    )
    plt.ylabel("Original Labels")
    plt.xlabel("Labels after Classification")
    plt.title("Confusion Matrix")


def generate_precision_dataframe(
    folder_path, test_label_list, translated_model
):
    """
    Function that returns a generated a dataframe of all the precision values
    evaluated for a true and predicted labels
    after classifiaction analysis on a dataset.
    """
    # Generate list of class names and indices
    class_names = Image_Loader.show_classes(
        folder_path=folder_path, dict_values=False
    )

    # Calculate precision values
    precision_list = precision_score(
        test_label_list.label, translated_model, average=None
    )

    # Generate precision dataframe
    precision_df = pd.DataFrame(
        data=precision_list.reshape(1, 14), columns=class_names
    )

    return precision_df
