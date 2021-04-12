import os
import scipy
import torch
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve, precision_recall_curve


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


class Model(torch.nn.Module):
    def __init__(self, layer_1, layer_2, layer_3):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(9, layer_1)
        self.lin2 = torch.nn.Linear(layer_1, layer_2)
        self.lin3 = torch.nn.Linear(layer_2, layer_3)
        self.lin4 = torch.nn.Linear(layer_3, 1)
        self.selu = torch.nn.SELU()

    def forward(self, x):
        x = self.selu(self.lin1(x))
        x = self.selu(self.lin2(x))
        x = self.selu(self.lin3(x))
        x = self.lin4(x)
        return x


def main():
    x = np.load(os.getcwd()+'/data/x.npy')
    y = np.load(os.getcwd()+'/data/y.npy')
    skf = StratifiedKFold(n_splits=10)
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
        imputer = IterativeImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x[train_index]))
        x_test = scaler.transform(imputer.transform(x[test_index]))
        x_train, y_train = torch.from_numpy(x_train).float().to('cuda:0'), torch.from_numpy(y_train).float().to('cuda:0')
        x_test = torch.from_numpy(x_test).float().to('cuda:0')
        model = Model(197, 198, 112)
        no_epochs = 127
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.80], device='cuda:0'))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03104, weight_decay=0.01043, momentum=0.4204,
                                    nesterov=True)
        model.train()
        model.to('cuda:0')
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train.view(-1, 1))
            loss.backward()
            optimizer.step()
        model.eval()
        y_pred.append(torch.sigmoid(model(x_test)).detach().cpu().numpy())
        y_true.append(y_test)
    for fold in range(len(y_pred)):
        tn, fp, fn, tp = confusion_matrix(y_true[fold], np.round(y_pred[fold])).ravel()
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision.append(tp / (tp + fp))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        roc_auc.append(roc_auc_score(y_true[fold], np.round(y_pred[fold])))
        prc_auc.append(average_precision_score(y_true[fold], np.round(y_pred[fold])))
        balanced_acc.append(balanced_accuracy_score(y_true[fold], np.round(y_pred[fold])))
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
    fpr, tpr, thresholds = roc_curve(np.hstack(y_true), np.vstack(y_pred))
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    precision, recall, thresholds = precision_recall_curve(np.hstack(y_true), np.vstack(y_pred))
    plt.plot(precision, recall)
    plt.title('PRC Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
    prob_true, prob_pred = calibration_curve(np.hstack(y_true), np.vstack(y_pred))
    plt.plot(prob_true, prob_pred)
    plt.title('Calibration Curve')
    plt.xlabel('Prob True')
    plt.ylabel('Prob Pred')
    plt.show()


if __name__ == '__main__':
    main()
