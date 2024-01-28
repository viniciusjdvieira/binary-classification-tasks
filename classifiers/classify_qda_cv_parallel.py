from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
import numpy as np

def get_classifier(params):
    ''' 
    This function employs a simple classification with QDA and
    cross validation (cv)
    '''
    X, y, Nfolds = params

    parallelism = False

    if parallelism:
        print("sorry!\n We'll try to use parallelism later.")
        return (0,) * 14

    else:
        # model QDA
        qda = QuadraticDiscriminantAnalysis()
        
        # kFold instances
        kf = KFold(n_splits=Nfolds, shuffle=True, random_state=42)

        # Results
        accuracy_scores = []
        specificity_scores = []
        sensitivity_scores = []
        auc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Training model
            qda.fit(X_train, y_train)
            
            # predictions
            y_pred = qda.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # values from cm
            TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
            
            # Performance mtrics
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            y_prob = qda.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            precision = precision_score(y_test, y_pred, zero_division = 0)
            recall =  recall_score(y_test, y_pred)
            f1_s =  f1_score(y_test, y_pred)
            
            accuracy_scores.append(accuracy)
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
            auc_scores.append(roc_auc)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_s)
        
        # average value from performance metrics
        # the standard deviation is normalized as the pattern error for inference,
        # because the sample size (Nfolds) < 30
        mean_accuracy = np.nanmean(accuracy_scores)
        std_accuracy = np.nanstd(accuracy_scores) / np.sqrt(Nfolds)
        
        mean_sensitivity = np.nanmean(sensitivity_scores)
        std_sensitivity = np.nanstd(sensitivity_scores) / np.sqrt(Nfolds)
        
        mean_specificity = np.nanmean(specificity_scores)
        std_specificity = np.nanstd(specificity_scores) / np.sqrt(Nfolds)

        mean_auc = np.nanmean(auc_scores)
        std_auc = np.nanstd(auc_scores) / np.sqrt(Nfolds)

        mean_precision = np.nanmean(precision_scores)
        std_precision = np.nanstd(precision_scores) / np.sqrt(Nfolds)

        mean_recall = np.nanmean(recall_scores)
        std_recall = np.nanstd(recall_scores) / np.sqrt(Nfolds)

        mean_f1 = np.nanmean(f1_scores)
        std_f1 = np.nanstd(f1_scores) / np.sqrt(Nfolds)

        dict_qda = {
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
            'std_f1' : std_f1 * 100
        }

        return dict_qda