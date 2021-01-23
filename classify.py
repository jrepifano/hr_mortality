import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve, precision_recall_curve


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def main():
    x = np.load(os.getcwd()+'/data/x.npy')
    y = np.load(os.getcwd()+'/data/y.npy')
    skf = StratifiedKFold(n_splits=10)
    coefs = []
    y_pred = []
    y_true = []
    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    roc_auc = []
    prc_auc = []
    balanced_acc = []
    for train_index, test_index in skf.split(x, y):
        y_train, y_test = y[train_index], y[test_index]
        imputer = SimpleImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x[train_index]))
        x_test = scaler.transform(imputer.transform(x[test_index]))
        lgr = LogisticRegression(class_weight='balanced').fit(x_train, y_train)
        coefs.append(lgr.coef_)
        y_pred.append(lgr.predict_proba(x_test))
        y_true.append(y_test)
    for fold in range(len(y_pred)):
        tn, fp, fn, tp = confusion_matrix(y_true[fold], np.round(y_pred[fold][:, 1])).ravel()
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision.append(tp / (tp + fp))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        roc_auc.append(roc_auc_score(y_true[fold], np.round(y_pred[fold][:, 1])))
        prc_auc.append(average_precision_score(y_true[fold], np.round(y_pred[fold][:, 1])))
        balanced_acc.append(balanced_accuracy_score(y_true[fold], np.round(y_pred[fold][:, 1])))
    mean, confidence_interval = mean_confidence_interval(accuracy)
    print('Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(precision)
    print('Precision Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(sensitivity)
    print('Sensitivity Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(specificity)
    print('Specificity Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(roc_auc)
    print('ROC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(prc_auc)
    print('PRC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(balanced_acc)
    print('Balanced Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    fpr, tpr, thresholds = roc_curve(np.hstack(y_true), np.vstack(y_pred)[:, 1])
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    precision, recall, thresholds = precision_recall_curve(np.hstack(y_true), np.vstack(y_pred)[:, 1])
    plt.plot(precision, recall)
    plt.title('PRC Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
    prob_true, prob_pred = calibration_curve(np.hstack(y_true), np.vstack(y_pred)[:, 1])
    plt.plot(prob_true, prob_pred)
    plt.title('Calibration Curve')
    plt.xlabel('Prob True')
    plt.ylabel('Prob Pred')
    plt.show()


if __name__ == '__main__':
    main()
