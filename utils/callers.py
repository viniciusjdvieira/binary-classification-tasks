import os
import itertools as its
from utils import process_files as pf 
from utils import clf_individual as ci 
from utils import clf_combos as cc

def call_individual(df, n_metrics,
                    labels_clf, 
                    classifier, 
                    n_folds, 
                    parallelism,
                    fine_tuning_svm,
                    kernel_svm,
                    c_value, 
                    folder):
    metrics = tuple((df.keys().to_list()[-n_metrics:]))
    label_clf = labels_clf
    clf_opt = classifier
    par_opt = parallelism
    params = (n_metrics, 
              metrics, 
              label_clf, 
              df, 
              clf_opt, 
              n_folds, 
              par_opt, 
              c_value, 
              kernel_svm, 
              fine_tuning_svm)

    result_df = ci.classify_individual(params)
    results_folder = pf.output_folder(folder, clf_opt)
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    pf.save_result(result_df, results_folder, "individual", n_metrics, clf_opt)
    print("Results were put in the path:\n", results_folder, "\n")


def call_combinations(df, 
                      n_metrics,
                      combos,
                      labels_clf, 
                      classifier, 
                      n_folds, 
                      parallelism, 
                      fine_tuning_svm, 
                      kernel_svm, 
                      c_value,
                      folder,
                      save_all_results,
                      save_summary_file):
    
    metrics = tuple((df.keys().to_list()[-n_metrics:]))

    if combos == 'all':
        vector_combos = [i for i in range(2, n_metrics+1)]
    else:
        vector_combos = combos 
   
    for metric_idx in vector_combos:
        
        combos_metrics = tuple(its.combinations(metrics,metric_idx))

        label_clf = labels_clf
        clf_opt = classifier
        par_opt = parallelism
        params = (n_metrics, 
                  combos_metrics, 
                  label_clf, 
                  df, 
                  clf_opt, 
                  n_folds, 
                  par_opt, 
                  c_value, 
                  kernel_svm, 
                  fine_tuning_svm)
        
        result_df = cc.classify_combos(params)
        print("===> Refult dataframe for combinations of ", metric_idx, " metrics:")
        print(result_df)

        if save_all_results:
            results_folder = pf.output_folder(folder, clf_opt)
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)
            pf.save_result(result_df, results_folder, "combo", metric_idx, clf_opt)
            print("Results were put in the path:\n", results_folder, "\n")

    if save_summary_file:
        if not os.path.isdir(results_folder):
            print("Check if you have all files in ", results_folder)
        pf.summary_results(results_folder, classifier)
        print("The summarized results (THE BEST ONES) were put in the path:\n", results_folder, "\n")