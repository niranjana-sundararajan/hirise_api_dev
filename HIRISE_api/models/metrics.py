from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, mutual_info_score, rand_score, completeness_score, homogeneity_score, v_measure_score, balanced_accuracy_score

def calculate_metrics(model  , labels, verbose = False ):

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
  return [rs,ars, mis,nmis,amis,bas,cs,hs,vs]