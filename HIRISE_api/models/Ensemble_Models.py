import six
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import std
from numpy import mean
sys.modules['sklearn.externals.six'] = six  # noqa
from mlxtend.classifier import StackingClassifier



# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

if __package__ is None or __package__ == "":
    # uses current directory visibility
    import utils
else:
    from . import utils


# ---------------------------------------------------------------------------
#  WRAPPING CLASSES FOR INPUTS INTO ENSEMBLE MODELS
# ---------------------------------------------------------------------------


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    def predict(self, X):
        return self.fit_predict(X)


class OpticsWrapper(OPTICS):
    def predict(self, X):
        return self.fit_predict(X)


class DBSCANWrapper(DBSCAN):
    def predict(self, X):
        return self.fit_predict(X)


class HDBSCANWrapper(HDBSCAN):
    def predict(self, X):
        return self.fit_predict(X)


# ---------------------------------------------------------------------------
def get_stacking(discovery=False, all_models=True):
    """
    Function that stacks specified models together as an input to the
    ensemble model.
    """
    if discovery:
        affinity = AffinityPropagation(damping=0.7)
        affinity._estimator_type = "classifier"
        optics = OpticsWrapper(eps=0.5, min_samples=5)
        optics._estimator_type = "classifier"
        estimators = [affinity, optics]
        meta_learner = AffinityPropagation(damping=0.7)
        meta_learner._estimator_type = "classifier"
    else:
        ac = AgglomerativeClusteringWrapper(n_clusters=14)
        ac._estimator_type = "classifier"
        K_Means = KMeans(14, random_state=0)
        K_Means._estimator_type = "classifier"
        birch = Birch(threshold=0.2, n_clusters=14)
        birch._estimator_type = "classifier"
        estimators = [K_Means, ac, birch]
        meta_learner = AgglomerativeClusteringWrapper(n_clusters=14)
        meta_learner._estimator_type = "classifier"

    # define meta learner model
    meta_learner = KMeans(14, random_state=0)
    meta_learner._estimator_type = "classifier"
    model = StackingClassifier(
        classifiers=estimators, meta_classifier=meta_learner
    )
    return model


# ---------------------------------------------------------------------------
def get_models(discovery=False):
    """
    Function that defines specified models as an input to the ensemble model.
    """
    models = dict()
    if discovery:
        models["dbscan"] = DBSCANWrapper(eps=0.5, min_samples=5)
        models["hdbscan"] = HDBSCANWrapper(
            min_samples=5, gen_min_span_tree=True
        )
        models["affinity"] = AffinityPropagation(damping=0.7)
        models["optics"] = OpticsWrapper(eps=0.7, min_samples=9)
        models["stacking"] = get_stacking()
    else:
        models["ac"] = AgglomerativeClusteringWrapper(
            n_clusters=14, compute_full_tree=True, linkage="ward"
        )
        models["km"] = KMeans(14, random_state=0)
        models["birch"] = Birch(threshold=0.2, n_clusters=14)
        models["stacking"] = get_stacking(discovery=discovery)
    return models


# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    translation_dataframe,
    X,
    y,
    classification=False,
    clustering_model=None,
    dim_reduction_technique=None,
    transfer_learning_model=None,
    scoring_measure="v_measure_score",
):
    """
    Function that uses cross-validation and evalutes the stacking model.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    if classification:
        translation_values = translation_dataframe.TRANSLATION_LIST[
            translation_dataframe["CLUSTERING_MODEL"]
            == clustering_model & translation_dataframe["DIM_RED"]
            == dim_reduction_technique
            & translation_dataframe["TRANSFER_MODEL"]
            == transfer_learning_model
        ]

        print(utils.translate_labels(
            translation_list=list(translation_values), model_results=model
        ))

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=scoring_measure,
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    return scores


def ensemble_model(
    encoded_data,
    labels,
    translation_dataframe,
    transfer_learning_model=None,
    dim_reduction_technique=None,
    clustering_model=None,
    classification=False,
    plot=False,
    verbose=False,
):
    if isinstance(encoded_data, list):
        X_test_list = [StandardScaler().fit_transform(i) for i in encoded_data]
    else:
        X_test_list = [StandardScaler().fit_transform(encoded_data)]
    y_test = labels.label
    # get the models to evaluate
    models = get_models(discovery=True)
    # evaluate the models and store results
    results, names = list(), list()
    for X_test in X_test_list[3:]:
        for name, model in models.items():
            scores = evaluate_model(
                model=model,
                X=X_test,
                y=y_test,
                translation_dataframe=translation_dataframe,
                classification=classification,
                clustering_model=clustering_model,
                dim_reduction_technique=dim_reduction_technique,
                transfer_learning_model=transfer_learning_model,
            )
            results.append(scores)
            names.append(name)
            if verbose:
                print(
                    "%s %.3f (%.3f) mean (std): "
                    % (name, mean(scores), std(scores))
                )

    if plot:
        plt.figure(figsize=(30, 10))
        plt.boxplot(results, labels=names, showmeans=True)
        plt.show()
