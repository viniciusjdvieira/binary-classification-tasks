import numpy as np
import pandas as pd
from classifiers import classify_lda_cv, classify_qda_cv, classify_qda_cv_parallel, classify_lda_cv_parallel, classify_svm_cv, classify_svm_ft
from multiprocessing import Pool, cpu_count

def classify_individual(params):
    '''
    This function mades an individual classification

    params (tuple):
    - amount of individual metrics
    - tuple with the df.keys() with metrics' names
    - label with y values
    - dataframe
    - classifier option
    - n_folds
    - parallelism option
    - c_value for SVM
    - SVM kernel
    - fine tuning option for SVM
    '''
    fine_tuning_svm = params[-1]
    kernel_svm = params[-2]
    c_svm = params[-3]
    parallelism_opt = params[-4]
    n_folds = params[-5]
    classifier_opt = params[-6]
    df = params[-7]
    labels = params[-8]
    amnt = params[0]    
    metrics_tuple = params[1]

    if parallelism_opt:
        print("---> Here we have parallelism ...")
        indv_metric = []
        clf_info = []
        
        if classifier_opt == 'qda':
            params = []
            for metric in metrics_tuple:
                X = df[metric].astype(float)
                y = df[labels].astype(int)
                
                X = np.array(X).reshape(-1,1)
                y = np.array(y)

                params.append((X,y,n_folds))
                indv_metric.append(metric)
                clf_info.append(classifier_opt)

            n_cpus = cpu_count()
            batches = len(metrics_tuple) // n_cpus
            batches = max(1, batches)

            with Pool(processes=n_cpus) as pool:
                result = pool.map(classify_qda_cv_parallel.get_classifier, params, chunksize=batches)
                output = [x for x in result]

            pool.close()
            pool.join()

        elif classifier_opt == 'lda':
            params = []
            for metric in metrics_tuple:
                X = df[metric].astype(float)
                y = df[labels].astype(int)
                
                X = np.array(X).reshape(-1,1)
                y = np.array(y)

                params.append((X,y,n_folds))
                indv_metric.append(metric)
                clf_info.append(classifier_opt)

            n_cpus = cpu_count()
            batches = len(metrics_tuple) // n_cpus
            batches = max(1, batches)

            with Pool(processes=n_cpus) as pool:
                result = pool.map(classify_qda_cv_parallel.get_classifier, params, chunksize=batches)
                output = [x for x in result]

            pool.close()
            pool.join()

        elif classifier_opt == 'svm':
            print("Use SVM without parallelism.")


    else:
        print("---> Here we don't have parallelism ...")
        output = []
        indv_metric = []
        clf_info = []
        if classifier_opt == 'qda':
            for metric in metrics_tuple:
                X = df[metric].astype(float)
                y = df[labels].astype(int)
                
                X = np.array(X).reshape(-1,1)
                y = np.array(y)

                indv_metric.append(metric)
                clf_info.append(classifier_opt)
                output.append(classify_qda_cv.get_classifier(X,y,Nfolds=n_folds))

        elif classifier_opt == 'lda':
            for metric in metrics_tuple:
                X = df[metric].astype(float)
                y = df[labels].astype(int)
                
                X = np.array(X).reshape(-1,1)
                y = np.array(y)

                indv_metric.append(metric)
                clf_info.append(classifier_opt)
                output.append(classify_lda_cv.get_classifier(X,y,Nfolds=n_folds))

        elif classifier_opt == 'svm':
            if not fine_tuning_svm:
                for metric in metrics_tuple:
                    X = df[metric].astype(float)
                    y = df[labels].astype(int)
                
                    X = np.array(X).reshape(-1,1)
                    y = np.array(y)

                    indv_metric.append(metric)
                    clf_info.append(classifier_opt)
                    output.append(classify_svm_cv.get_classifier(X, 
                                                                 y, 
                                                                 kernel_param=kernel_svm, 
                                                                 C_param=c_svm, 
                                                                 Nfolds=n_folds))
            else:
                for metric in metrics_tuple:
                    X = df[metric].astype(float)
                    y = df[labels].astype(int)
                
                    X = np.array(X).reshape(-1,1)
                    y = np.array(y)

                    indv_metric.append(metric)
                    clf_info.append(classifier_opt)
                    output.append(classify_svm_ft.get_classifier(X, 
                                                                 y, 
                                                                 Nfolds=n_folds))


            
    df_results = pd.DataFrame({"metric":indv_metric, "info_clf":clf_info, "res": output})
    df_results = pd.concat([df_results, pd.json_normalize(df_results['res'])], axis=1)
    df_results = df_results.drop('res', axis=1)
    df_results = df_results.sort_values(by='mean_accuracy', ascending=False)
    
    return df_results