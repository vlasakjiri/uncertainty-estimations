import numpy as np
import torch
from scipy.stats import entropy


class Progress:
    def __init__(self) -> None:
        self.probs = []
        self.predictions = np.array([])
        self.labels = np.array([])
        self.max_probs = np.array([])
        self.confidences = np.array([])
        self.dropout_outputs = []
        self.dropout_variances = np.array([])
        self.dropout_predictions = np.array([])

    def update(self, preds, labels, probs):
        self.probs.append(probs.numpy())
        self.predictions = np.append(
            self.predictions, preds.numpy())
        self.labels = np.append(
            self.labels, labels.data.numpy())
        self.max_probs = np.append(self.max_probs, torch.max(
            probs, dim=1)[0].detach().numpy())
        self.confidences = np.append(self.confidences,
                                     1 - normalized_entropy(probs.detach().numpy(), axis=1))

    def update_mcd(self, mc_means, mc_var):
        mc_predictions = mc_means.argmax(axis=-1)
        self.dropout_outputs.append(mc_means)
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
