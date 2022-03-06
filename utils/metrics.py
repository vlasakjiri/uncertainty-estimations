import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import sklearn.metrics


class model_metrics:
    def __init__(self) -> None:
        self.acc = []
        self.ece = []
        self.confs = []
        self.brier = []
        self.auroc = []
        self.aupr = []
        self.strengths = []

    def __str__(self) -> str:
        return str(vars(self))


def compute_model_stats(correct, max_probs, label):
    fpr, tpr, _ = sklearn.metrics.roc_curve(correct, max_probs)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    prec, recall, _ = sklearn.metrics.precision_recall_curve(
        correct, max_probs)
    aupr = sklearn.metrics.auc(recall, prec)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auroc": roc_auc,
        "prec": prec,
        "recall": recall,
        "aupr": aupr,
        "label": label
    }


def update_model_metrics(progress, probs, max_probs, predictions, labels, bins, m: model_metrics(), strength):
    accs, errors, counts = compute_calibration_metrics(
        predictions, labels, max_probs, np.argsort(max_probs), bins)
    ece = np.average(errors, weights=counts) * 100
    mean_acc = np.average(accs, weights=counts)
    mean_confidence = np.mean(max_probs)

    y_true = torch.nn.functional.one_hot(torch.tensor(labels.astype("long")))
    brier = compute_brier_score_avg(probs, y_true)

    correct = predictions == labels
    curve = compute_model_stats(correct, max_probs, "dasd")
    m.acc.append(mean_acc)
    m.ece.append(ece)
    m.confs.append(mean_confidence)
    m.brier.append(brier)
    m.auroc.append(curve["auroc"])
    m.aupr.append(curve["aupr"])
    m.strengths.append(strength)


class Progress:
    def __init__(self) -> None:
        self.logits = []
        self.probs = []
        self.predictions = []
        self.labels = np.array([])
        self.max_probs = np.array([])
        self.confidences = np.array([])
        self.dropout_nlls = []
        self.dropout_outputs = []
        self.dropout_variances = np.array([])
        self.dropout_predictions = np.array([])

    def update(self, preds, labels, probs, logits):
        self.logits.append(logits.numpy())
        self.probs.append(probs.numpy())
        self.predictions.append(preds.numpy())
        self.labels = np.append(
            self.labels, labels.data.numpy())
        self.max_probs = np.append(self.max_probs, torch.max(
            probs, dim=1)[0].detach().numpy())
        self.confidences = np.append(self.confidences,
                                     1 - normalized_entropy(probs.detach().numpy(), axis=1))

    def update_mcd(self, mc_means, mc_var, nll):
        mc_predictions = mc_means.argmax(axis=-1)
        self.dropout_nlls.append(nll.numpy())
        self.dropout_outputs.append(mc_means.numpy())
        self.dropout_predictions = np.append(
            self.dropout_predictions, mc_predictions)
        self.dropout_variances = np.append(
            self.dropout_variances, mc_var)


def __str__(self) -> str:
    return f"Predictions: {self.predictions}\nLabels: {self.labels}\nMax probs: {self.max_probs}\n Confidences: {self.confidences}"


def normalized_entropy(probs, **kwargs):
    num_classes = probs.shape[-1]
    return entropy(probs, **kwargs)/np.log(num_classes)


def roc_stat(labels, predictions, step=10):
    accs = []
    for _ in range(len(labels)//step):
        accs.append((labels == predictions).sum()/len(labels))
        labels = labels[step:]
        predictions = predictions[step:]
    return accs


def compute_brier_score_avg(y_pred, y_true):
    """Brier score implementation follows
    https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
    The lower the Brier score is for a set of predictions, the better the predictions are calibrated."""

    brier_score = torch.mean((y_true-y_pred)**2, 1)
    return brier_score.mean().item()


def compute_calibration_metrics(predictions, labels, confidences, idx, bins):
    accs = []
    errors = []
    counts = []
    confidences = confidences[idx]
    labels = labels[idx]
    predictions = predictions[idx]
    for i, bin in enumerate(bins):
        idx = np.argwhere((confidences >= bin - 0.05) &
                          (confidences <= bin + 0.05))
        if len(idx) > 20:
            mean_acc = (predictions[idx] == labels[idx]).sum() / len(idx)
            err = np.abs(mean_acc-bin)
        else:
            mean_acc = 0
            err = 0
        accs.append(mean_acc)
        errors.append(err)
        counts.append(len(idx))
    return accs, errors, counts


def iou(preds, labels, num_classes):
    preds = F.one_hot(preds, num_classes)[..., 1:]
    labels = F.one_hot(labels, num_classes)[..., 1:]
    intersection = (preds & labels).sum()
    union = (preds | labels).sum()
    iou = (intersection) / (union)

    return iou.nanmean()


def mean_class_iou(preds, labels, num_classes):
    preds = F.one_hot(preds, num_classes)
    labels = F.one_hot(labels, num_classes)
    intersection = (preds & labels).sum((0, 1, 2))
    union = (preds | labels).sum((0, 1, 2))
    iou = (intersection) / (union)

    return iou.nanmean()
