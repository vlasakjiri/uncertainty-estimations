
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt
import numpy as np
import utils.metrics


def plot_uncertainties(progress):
    correct = progress.predictions == progress.labels
    mc_dropout_confidence = 1 - progress.dropout_variances / \
        progress.dropout_variances.max()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.violinplot([progress.confidences, progress.max_probs, mc_dropout_confidence,
                   progress.confidences[correct], progress.max_probs[correct], mc_dropout_confidence[correct],
                   progress.confidences[~correct], progress.max_probs[~correct], mc_dropout_confidence[~correct]])

    print(
        f"All predictions mean confidence: {np.mean(progress.confidences)}, "
        f"Prob: {np.mean(progress.max_probs)}, "
        f"Var: {mc_dropout_confidence.mean()}")
    print(
        f"Correct predictions mean confidence: {np.mean(progress.confidences[correct])}, "
        f"Prob: {np.mean(progress.max_probs[correct])}, "
        f"Var: {mc_dropout_confidence[correct].mean()}")
    print(
        f"Incorrect predictions mean confidence: {np.mean(progress.confidences[~correct])}, "
        f"Prob: {np.mean(progress.max_probs[~correct])}, "
        f"Var: {mc_dropout_confidence[~correct].mean()}")


def samples_removed_vs_acc(label_idx_list, labels_in, preds_in, dropout_preds_in):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    for label, idx in label_idx_list:
        labels = labels_in[idx]
        predictions = dropout_preds_in[
            idx] if "dropout" in label.lower() else preds_in[idx]
        accs = utils.metrics.roc_stat(labels, predictions, step=10)
        ax.plot(np.linspace(0, 100, len(accs)), accs, label=label)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()


def calibration_graph(label_idx_list, labels_in, preds_in, dropout_preds_in):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    bins = np.linspace(0, 1, num=10)
    for label, sort, idx in label_idx_list:
        sort = sort[idx]
        labels = labels_in[idx]
        predictions = dropout_preds_in[
            idx] if "dropout" in label.lower() else preds_in[idx]
        inds = np.digitize(sort, bins)
        accs = []
        for i, bin in enumerate(bins):
            idx = np.argwhere(inds == i)
            acc = (predictions[idx] == labels[idx]).sum() / \
                len(idx) if len(idx) > 5 else 0
            accs.append(acc)
        ax.plot(bins, accs, label=label)
    ax.plot(bins, np.linspace(0, 1, len(bins)))
    ax.set_ylim([0, 1])
    ax.legend()
