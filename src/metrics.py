import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, f1_score, roc_curve, \
    precision_recall_curve
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index


def eval_binary(model, dataloader, criterion, device):
    avg_loss, scores, avg_precision, auroc, cm = [], [], 0, 0, [0,0,0,0]
    metrics = {}
    model.eval()
    with torch.no_grad():
        for _, (drug, target, label) in enumerate(tqdm(dataloader, position=0, desc='Validation/Testing')):
            drug = drug.to(device)
            target = target.to(device)
            label = label.float().to(device)
            score = model(drug, target)

            prob = torch.sigmoid(score).to('cpu').data.numpy()
            score = score.squeeze(1)
            loss = criterion(score, label)
            avg_loss.append(loss)
            label = label.to('cpu').data.numpy()
            # label = [1 if l == 1 else 0 for l in label]
            for idx in range(len(label)):
                scores.append([(int(label[idx]), prob[idx][0])])
    scores = np.array(scores)
    scores = np.squeeze(scores)
    threshold = optimal_cutoff(scores[:, 0], scores[:, 1])
    pred = scores[:, 1] > threshold
    pred = np.array(pred, dtype=np.int32)
    scores = np.column_stack([scores, pred])


    assert scores.shape[1] == 3
    avg_loss = np.mean([loss.item() for loss in avg_loss])

    avg_precision = average_precision_score(scores[:,0], scores[:,1])
    auroc = roc_auc_score(scores[:,0], scores[:,1])
    cm = confusion_matrix(scores[:,0], scores[:,2]).ravel()

    f1 = f1_score(scores[:,0], scores[:,2])

    metrics['scores'] = scores
    metrics['avg_loss'] = avg_loss
    metrics['avg_precision'] = avg_precision
    metrics['auroc'] = auroc
    metrics['tp'] = cm[0]
    metrics['fp'] = cm[1]
    metrics['fn'] = cm[2]
    metrics['tn'] = cm[3]
    metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])
    metrics['sensitivity'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
    metrics['fnr'] = 1 - metrics['sensitivity']
    metrics['fpr'] = 1 - metrics['specificity']
    metrics['f1'] = f1
    return metrics

def optimal_cutoff(y_true, y_predicted):
    '''
    The following implementation is based on youden's index. It is the maximum
    vertical distance between ROC curve and diagonal line. The idea is to maximize
    the difference between True Positive and False Positive.
    J = Sensitivity - (1-Specificity)
    Other optimal cutoff points: https://www.listendata.com/2015/03/sas-calculating-optimal-predicted.html
    However, it is suggested that Youdens Index works well. http://www.medicalbiostatistics.com/roccurve.pdf
    :return: optimal point (float)
    '''
    # fpr, tpr, threshold = roc_curve(y_true, y_predicted)
    # i = np.arange(len(tpr))
    # roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    # roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]
    #
    # return roc_t['threshold'].values[0]

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_predicted, pos_label=1)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    return threshold

def eval_mse_model(model, dataloader, criterion, device):
    scores, labels, losses  = [], [], []
    metrics = {}

    model.eval()
    with torch.no_grad():
        #for _, (batch) in enumerate(tqdm(dataloader, position=0, desc='Validation/Testing')):
        for _, (batch) in enumerate(dataloader):
            batch = batch.to(device)
            score = model(batch)
            label = batch.y.float().to(device)
            loss = criterion(score, label)
            losses.append(loss)
            scores.append(score.to('cpu').data.numpy())
            labels.append(label.to('cpu').data.numpy())
    metrics['avg_loss'] = np.mean([loss.item() for loss in losses])
    metrics['total_loss'] = np.sum([loss.item() for loss in losses])
    scores = np.array([score for batch_scores in scores for score in batch_scores]).squeeze()
    labels = np.array([label for batch_labels in labels for label in batch_labels ]).squeeze()
    metrics['mse'] = mean_squared_error(labels, scores)
    metrics['cor'], metrics['cor_pval'] = pearsonr(labels, scores)
    metrics['concordance_score'] = concordance_index(labels, scores)

    return metrics