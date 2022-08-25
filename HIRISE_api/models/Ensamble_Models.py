# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.estimator_checks import check_estimator
from mlxtend.classifier import StackingClassifier
from sklearn.svm import LinearSVC
import six
import sys
sys.modules['sklearn.externals.six'] = six

def get_models():
 models = dict()
 for name, mod in zip(models_name_list_string,translated_models_list):
    models[name] = mod
 return models


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    def predict(self,X):
      return self.fit_predict(X)
class OpticsWrapper(OPTICS):
    def predict(self,X):
      return self.fit_predict(X)
class DBSCANWrapper(DBSCAN):
    def predict(self,X):
      return self.fit_predict(X)
class HDBSCANWrapper(HDBSCAN):
    def predict(self,X):
      return self.fit_predict(X)

# sys.modules['sklearn.externals.six'] = six

def get_stacking(dicovery = False, all_models = True):
  if dicovery:
      affinity = AffinityPropagation(damping=0.7)
      affinity._estimator_type = "classifier"
      optics = OpticsWrapper(eps=0.5 ,min_samples=5)
      optics._estimator_type = "classifier"
      # dbscan = DBSCANWrapper(eps = 0.5, min_samples = 5)
      # hdbscan = HDBSCANWrapper(min_samples=5, gen_min_span_tree=True)
      mean_shift = MeanShift()
      # estimators=  [affinity, optics, dbscan,hdbscan,mean_shift]
      estimators=  [affinity, optics]
      meta_learner = AffinityPropagation(damping=0.7)
      meta_learner._estimator_type = 'classifier'
  else:
    ac = AgglomerativeClusteringWrapper(n_clusters=14)
    ac._estimator_type = "classifier"
    K_Means = KMeans( 14,random_state=0)
    K_Means._estimator_type = "classifier"
    birch = Birch(threshold=0.2, n_clusters=14)
    birch._estimator_type = "classifier"
    estimators=  [K_Means, ac, birch]
    meta_learner =AgglomerativeClusteringWrapper(n_clusters=14)
    meta_learner._estimator_type = 'classifier'

# define meta learner model
  meta_learner = KMeans(14,random_state=0)
  meta_learner._estimator_type = 'classifier'
# define the stacking ensemble
  model = StackingClassifier(classifiers=estimators, meta_classifier=meta_learner)
  return model

# get a list of models to evaluate
def get_models(discovery = False):
  models = dict()
  if discovery:
    models['dbscan'] = DBSCANWrapper(eps = 0.5, min_samples = 5)
    models['hdbscan'] = HDBSCANWrapper(min_samples=5, gen_min_span_tree=True)
    models['affinity'] = AffinityPropagation(damping=0.7)
    models['optics'] = OpticsWrapper(eps=0.7 ,min_samples=9)
    # models['mean_shift'] = MeanShift()
    models['stacking'] = get_stacking()
  else:
    models['ac'] = AgglomerativeClusteringWrapper(n_clusters=14,compute_full_tree=True, linkage='ward')
    models['km'] = KMeans( 14,random_state=0)
    models['birch'] = Birch(threshold=0.2, n_clusters=14)
    models['stacking'] = get_stacking(dicovery = discovery)
  return models





# evaluate a give model using cross-validation
def evaluate_model(model, X, y, classification = False, clustering_model = None, dim_reduction_technique = None, transfer_learning_model = None):
  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
  if classification:
    translation_vlaues = translation_dataframe.TRANSLATION_LIST[translation_dataframe['CLUSTERING_MODEL'] == clustering_model & translation_dataframe['DIM_RED'] == dim_reduction_technique & translation_dataframe['TRANSFER_MODEL'] == transfer_learning_model ]
    translated_results = translate_labels(translation_list = list(translation_vlaues), model_results = model)

  scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=1, error_score='raise')
  return scores



def ensemble_model(encoded_data,labels,models, transfer_learning_model = None, dim_reduction_technique = None, clustering_model = None, classification = False, plot = False):
  if isinstance(encoded_data, list):
    X_test_list = [StandardScaler().fit_transform(i) for i in encoded_data]
  else:
     X_test_list = [StandardScaler().fit_transform(encoded_data)]
  y_test = labels.label
  # get the models to evaluate
  models = get_models(discovery = True)
  # evaluate the models and store results
  results, names = list(), list()
  for X_test in X_test_list[3:]:
    for name, model in models.items():
      # print(X_test.shape, y_test.shape)
      scores = evaluate_model(model, X_test, y_test)
      results.append(scores)
      names.append(name)
      print('%s %.3f (%.3f) mean (std): ' % (name, mean(scores), std(scores)))
  if classification:
    ...

  if plot:
    # plot model performance for comparison
    plt.figure(figsize=(30,10))
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()