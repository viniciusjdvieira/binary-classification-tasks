import os 
import sys 
import pandas as pd 
import numpy as np 
from utils import process_files, show_info, callers

if __name__ == '__main__':
    config_file_path = os.path.normpath(sys.argv[1])

    print("Reading config file ...")
    config_file = process_files.read_config_file(config_file_path)
    parameters = process_files.extract_params(config_file)

    classifier = parameters["classifier"]
    show_info.print_classifier(classifier)

    print("Loading dataset ...")
    dataset_path = parameters["dataset_path"]
    df = pd.read_csv(dataset_path)
    
    # Taking the other parameters
    n_metrics = parameters["number_of_metrics"]
    labels_clf = parameters["label_column"]
    parallelism = parameters["parallelism"]
    result_path = parameters["results_path"]
    n_folds = parameters["n_folds"]
    individual = parameters["individual"]
    combinations = parameters["combinations"]
    save_all_results = parameters["save_all_results"]
    save_summary_file = parameters["save_summary_file"]
    fine_tuning_svm = parameters["fine_tuning_svm"]
    kernel_svm = parameters["kernel_svm"]
    c_value_svm = parameters["c_value"]
    combos = parameters["combos"]

  
    if individual:
        print(">>> Running individual classification ...")
        callers.call_individual(df, 
                                n_metrics, 
                                labels_clf, 
                                classifier, 
                                n_folds, 
                                parallelism,
                                fine_tuning_svm,
                                kernel_svm,
                                c_value_svm, 
                                result_path)
    if combinations:
        print(">>> Running classification with combinations ...")
        callers.call_combinations(df, 
                                n_metrics, 
                                combos, 
                                labels_clf, 
                                classifier, 
                                n_folds, 
                                parallelism, 
                                fine_tuning_svm,
                                kernel_svm,
                                c_value_svm,
                                result_path, 
                                save_all_results, 
                                save_summary_file)
    
