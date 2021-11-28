
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt, widgets
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


def calibration_graph(label_idx_list, labels_in, preds_in, dropout_preds_in, num_bins=10):
    fig, axs = plt.subplots(2, len(label_idx_list), figsize=(14, 10))
    bins = np.linspace(0.05, 0.95, num=10)
    counts = []
    for j, (label, confidence, idx) in enumerate(label_idx_list):
        ax = axs[0][j]
        confidence = confidence[idx]
        labels = labels_in[idx]
        predictions = dropout_preds_in[
            idx] if "dropout" in label.lower() else preds_in[idx]
        inds = np.digitize(confidence, bins)
        accs = []
        errors = []
        counts.append([])
        for i, bin in enumerate(bins):
            idx = np.argwhere((confidence >= bin-0.05) &
                              (confidence <= bin+0.05))
            if len(idx) > 20:
                acc = (predictions[idx] == labels[idx]).sum() / len(idx)
                err = np.abs(acc-bin)
            else:
                acc = 0
                err = 0
            accs.append(acc)
            errors.append(err)
            counts[j].append(len(idx))
        ax.bar(bins, accs, 0.1, label="Outputs", edgecolor="black")
        ax.bar(bins, errors, 0.1, bottom=accs, label="Gap",
               edgecolor="black")

        ax.plot(np.linspace(0, 1, len(bins)),
                np.linspace(0, 1, len(bins)), "k--")
        ece = np.average(errors, weights=counts[j])*100
        counts[j] = np.asarray(counts[j])
        ax.annotate(f"ECE: {ece:.2f}%", (0, 0.8), fontsize=15)
        ax.set_title(label)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()
        axs[1][j].bar(np.linspace(0.05, 0.95, 10),
                      counts[j]/counts[j].sum(), 0.1, edgecolor="black")

        axs[1][j].set_ylim(0, 1)
        axs[1][j].axvline(confidence.mean(), c="r",
                          ls="--", label="Confidence")
        axs[1][j].axvline(np.average(accs, weights=counts[j]),
                          c="g", ls="--", label="Accuracy")

        axs[1][j].set_xlabel("Confidence")
        axs[1][j].set_ylabel(r"% of Samples")
        axs[1][j].legend()
    return counts