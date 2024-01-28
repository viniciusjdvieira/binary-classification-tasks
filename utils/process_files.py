import yaml
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

def read_config_file(filepath):
    '''
    With this funtcion, we are able to read the config file.
    '''
    try:
        with open(filepath) as file:
            config_file = yaml.safe_load(file)
        return config_file
    except FileNotFoundError:
        print(f"Configuration file {filepath} not found.")
        return None 
    except yaml.YAMLError as e:
        print(f"Error while loading the file {filepath}.")
        return None 


def extract_params(config):
    '''
    Funtcion to extract the parameters from the config file.
    '''
    if config:
        classifier = config.get('classifier')
        
        kfold = config.get('KFold')
        n_folds = kfold.get('n_folds')
        
        classification = config.get('classification')
        individual = classification.get('individual')
        combinations = classification.get('combinations')
        save_all_results = classification.get('save_all_results')
        save_summary_file = classification.get('save_summary_file')

        svm_config = config.get('svm_config')
        fine_tuning_svm = svm_config.get('fine_tuning_svm')
        kernel_svm = svm_config.get('kernel_svm')
        c_value = svm_config.get('C_value')
        
        filepaths = config.get('filepaths')
        dataset_path = filepaths.get('dataset_path')
        results_path = filepaths.get('results_path')
        
        dataset = config.get('dataset')
        label_column = dataset.get('label_column')
        number_of_metrics = dataset.get('number_of_metrics')
        combos = dataset.get('combos')

        parallelism = config.get('parallelism')
        
        return {
            'classifier': classifier,
            'n_folds': n_folds,
            'individual': individual,
            'combinations': combinations,
            'save_all_results': save_all_results,
            'save_summary_file': save_summary_file,
            'fine_tuning_svm': fine_tuning_svm,
            'kernel_svm': kernel_svm,
            'c_value': c_value,
            'dataset_path': dataset_path,
            'results_path': results_path,
            'label_column': label_column,
            'number_of_metrics': number_of_metrics,
            'combos': combos,
            'parallelism': parallelism
        }
    else:
        return {}


def output_folder(filepath, classifier):
    '''
    Function to define the folder where we put the results files.
    '''
    try:
        current_date = datetime.now()
        date_format = current_date.strftime("%Y%m%d")
        folder = os.path.join(filepath,classifier+"_"+date_format)

        return folder
    
    except:
        print("Error... Check your output folder.")
        return None 


def save_result(df, result_path, class_type, n_metrics, classifier):
    '''
    Function to save the results
    '''
    current_date = datetime.now()
    date_format = current_date.strftime("%Y%m%d")

    if class_type == "individual":
        file_path = result_path + os.sep + "classification_" + class_type + "_classifier_" + classifier + "_" + date_format + ".csv"
    else:
        file_path = result_path + os.sep + "classification_" + class_type + "_" + str(n_metrics) + "_metrics_" + "_classifier_" + classifier + "_" + date_format + ".csv"

    df.to_csv(file_path)
    return file_path


def count_metrics(x):
    '''
    Function to count the amount of metrics in a combination.
    '''
    try:
        return len(eval(x))
    except:
        return 1


def concat_dfs(list_of_files):
    '''
    Function to concat the best results in dataframes.
    '''
    f_rows = []
    for filei in list_of_files:
        df = pd.read_csv(filei)
        if "Unnamed: 0" in df.keys():
            df = df.drop('Unnamed: 0', axis=1)
        first_row = df.iloc[0]
        f_rows.append(first_row)
    result_df = pd.concat(f_rows, axis=1)
    result_df = result_df.T 
    result_df.reset_index(drop=True, inplace=True)
    result_df['amnt_metrics'] = result_df['metric'].apply(count_metrics)
    result_df = result_df.sort_values(by='mean_accuracy', ascending=False)
    return result_df


def summary_results(folder, classifier):
    '''
    Function to get a summary of the best results.
    '''
    if os.path.isdir(folder):
        list_files = []
        for root_path, in_path, files in os.walk(folder):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root_path, file)
                    list_files.append(os.path.normpath(full_path))

        current_date = datetime.now()
        date_format = current_date.strftime("%Y%m%d")

        results = concat_dfs(list_files)
        results.to_csv(folder + os.sep + "summary_classification_" + classifier + "_" + date_format + ".csv")

    else:
        print("Error. Check the file path with the results.")
