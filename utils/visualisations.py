
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt, widgets
import numpy as np
from sklearn import metrics
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


def calibration_graph(label_idx_list, labels_in, **kwargs):
    fig, axs = plt.subplots(2, len(label_idx_list),
                            squeeze=False, **kwargs)
    bins = np.linspace(0.05, 0.95, num=10)
    for j, (label, confidence, idx, predictions) in enumerate(label_idx_list):
        ax = axs[0][j]
        accs, errors, counts = utils.metrics.compute_calibration_metrics(
            predictions, labels_in, confidence, idx, bins)
        counts = np.asarray(counts)
        ece = np.average(errors, weights=counts) * 100
        mean_acc = np.average(accs, weights=counts)

        ax.bar(bins, accs, 0.1, label="Outputs", edgecolor="black")
        ax.bar(bins, errors, 0.1, bottom=accs, label="Gap",
               edgecolor="black")

        ax.plot(np.linspace(0, 1, len(bins)),
                np.linspace(0, 1, len(bins)), "k--")

        ax.annotate(f"ECE: {ece:.2f}%", (0, 0.7), fontsize=15)
        ax.set_title(label)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()
        axs[1][j].bar(np.linspace(0.05, 0.95, 10),
                      counts/counts.sum(), 0.1, edgecolor="black")

        axs[1][j].set_ylim(0, 1)

        mean_conf = confidence.mean()
        axs[1][j].axvline(mean_conf, c="r",
                          ls="--", label=f"Confidence = {mean_conf:.3f}")
        axs[1][j].axvline(mean_acc,
                          c="g", ls="--", label=f"Accuracy = {mean_acc:.3f}")

        axs[1][j].set_xlabel("Confidence")
        axs[1][j].set_ylabel(r"% of Samples")
        axs[1][j].legend()
    plt.tight_layout()
