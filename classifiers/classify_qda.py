from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix


def get_classifier(X_train, y_train, X_test, y_test):
    ''' 
    This function employs a simple classification with QDA
    '''
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred_qda = qda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_qda)

    cm = confusion_matrix(y_test, y_pred_qda)

    # Values from confusion matrix
    TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

    # Calcule a sensibilidade (recall) e a especificidade
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    y_prob_qda = qda.predict_proba(X_test)[:, 1]
    fpr_qda, tpr_qda, _ = roc_curve(y_test, y_prob_qda)
    roc_auc_qda = auc(fpr_qda, tpr_qda)

    precision = precision_score(y_test, y_pred_qda)
    recall =  recall_score(y_test, y_pred_qda)
    f1_s =  f1_score(y_test, y_pred_qda)

    return accuracy, sensitivity, specificity, roc_auc_qda, precision, recall, f1_s