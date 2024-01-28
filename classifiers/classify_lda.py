from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix


def get_classifier(X_train, y_train, X_test, y_test):
    ''' 
    This function employs a simple classification with LDA
    '''
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_lda)

    cm = confusion_matrix(y_test, y_pred_lda)

    # Values from confusion matrix
    TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

    # Calcule a sensibilidade (recall) e a especificidade
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    y_prob_lda = lda.predict_proba(X_test)[:, 1]
    fpr_lda, tpr_lda, _ = roc_curve(y_test, y_prob_lda)
    roc_auc_lda = auc(fpr_lda, tpr_lda)

    precision = precision_score(y_test, y_pred_lda)
    recall =  recall_score(y_test, y_pred_lda)
    f1_s =  f1_score(y_test, y_pred_lda)

    return accuracy, sensitivity, specificity, roc_auc_lda, precision, recall, f1_s