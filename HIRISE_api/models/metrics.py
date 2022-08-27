from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, \
    mutual_info_score, rand_score, completeness_score, homogeneity_score, v_measure_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

from preprocessing import Image_Loader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(model, labels, verbose=False):
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
    metrics_list = []
    for mod in model_list:
        metrics_list.append(calculate_metrics(model=mod, labels=label_list))
    metrics_dataframe = pd.DataFrame(metrics_list,
                                     columns=["RAND_SCORE", "ADJUSTED RAND SCORE", "MUTUAL INFO SCORE",
                                              "NORMALIZED MUTUAL INFO SCORE",
                                              "ADJUSTED MUTUAL INFO SCORE", "BALANCED ACCURACY SCORE",
                                              "COMPLETENESS SCORE",
                                              "HOMOGENEITY SCORE", "V SCORE"])
    return metrics_dataframe


def print_confusion_matrix(folder_path, test_labels, translated_values_array, fig_size=(15, 10)):
    dataset1 = Image_Loader.generate_dataset(folder_path)
    class_names = dataset1.dataset.class_to_idx
    cm = confusion_matrix(test_labels.label, translated_values_array)
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize=fig_size)
    sns.heatmap(df_cm, annot=True, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Original Labels")
    plt.xlabel("Labels after Classification")
    plt.title("Confusion Matrix")


def generate_precision_dataframe(folder_path, test_label_list, translated_model):
    class_names = Image_Loader.show_classes(folder_path=folder_path, dict_values=False)
    precision_list = precision_score(test_label_list.label, translated_model, average=None)
    precision_df = pd.DataFrame(data=precision_list.reshape(1, 14), columns=class_names)
    return precision_df
