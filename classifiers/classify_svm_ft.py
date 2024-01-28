from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

## auxiliary functions:
def custom_sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 1] / (cm[1, 1] + cm[1, 0])

def custom_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def custom_auc(model,X_test,y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return roc_auc
#####----------------------------------

def get_classifier(X, 
                   y,
                   Nfolds = 10, 
                   parallelism = False):
    ''' 
    This function employs a simple classification with SVM and
    cross validation (cv)
    '''

    # parameters for grid_search:
    param_grid = {'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  }

    if parallelism:
        print("sorry!\n We'll try to use parallelism later.")
        return (0,) * 14

    else:
        # model SVM
        model = SVC(probability=True)

        # metrics for the grid_search
        metrics = {'accuracy': make_scorer(accuracy_score),
                   'sensitivity': make_scorer(custom_sensitivity, greater_is_better=True),
                   'specificity': make_scorer(custom_specificity, greater_is_better=True),
                   'auc': make_scorer(roc_auc_score, needs_proba=True),
                   'f1': make_scorer(f1_score),
                   'recall': make_scorer(recall_score),
                   'precision': make_scorer(precision_score, zero_division = 0)
                   }
        
        grid_search = GridSearchCV(model, param_grid, cv=Nfolds, scoring=metrics, refit='accuracy')
                
        # Data split: train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # fit
        grid_search.fit(X_train, y_train)

        # results
        df_gscv = pd.DataFrame(grid_search.cv_results_)

        # sorted results by the best accuracy
        df_gscv_sorted = df_gscv.sort_values(by='mean_test_accuracy', ascending=False)

        # choosing the columns to take
        desired_columns = ['mean_test_accuracy', 
                           'std_test_accuracy', 
                           'mean_test_sensitivity', 
                           'std_test_sensitivity', 
                           'mean_test_specificity', 
                           'std_test_specificity', 
                           'mean_test_auc', 
                           'std_test_auc', 
                           'mean_test_precision', 
                           'std_test_precision', 
                           'mean_test_recall', 
                           'std_test_recall', 
                           'mean_test_f1', 
                           'std_test_recall', 
                           'params']
        
        list_of_results = df_gscv_sorted.loc[df_gscv_sorted.index[0], desired_columns]
        results = list_of_results.values

        # average value from performance metrics
        # the standard deviation is normalized as the pattern error for inference,
        # because the sample size (Nfolds) < 30
        mean_accuracy = results[0]
        std_accuracy = results[1] / np.sqrt(Nfolds)
        
        mean_sensitivity = results[2]
        std_sensitivity = results[3] / np.sqrt(Nfolds)
        
        mean_specificity = results[4]
        std_specificity = results[5] / np.sqrt(Nfolds)

        mean_auc = results[6]
        std_auc = results[7] / np.sqrt(Nfolds)

        mean_precision = results[8]
        std_precision = results[9] / np.sqrt(Nfolds)

        mean_recall = results[10]
        std_recall = results[11] / np.sqrt(Nfolds)

        mean_f1 = results[12]
        std_f1 = results[13] / np.sqrt(Nfolds)

        best_params = results[14]

        dict_svm = {
            'mean_accuracy' : mean_accuracy * 100,
            'std_accuracy' : std_accuracy * 100,
            'mean_sensitivity' : mean_sensitivity * 100,
            'std_sensitivity' : std_sensitivity * 100,
            'mean_specificity' : mean_specificity * 100,
            'std_specificity' : std_specificity * 100,
            'mean_auc' : mean_auc,
            'std_auc' : std_auc,
            'mean_precision' : mean_precision * 100,
            'std_precision' : std_precision * 100,
            'mean_recall' : mean_recall * 100,
            'std_recall' : std_recall * 100,
            'mean_f1' : mean_f1 * 100,
            'std_f1' : std_f1 * 100,
            'best_params': best_params
        }

        return dict_svm