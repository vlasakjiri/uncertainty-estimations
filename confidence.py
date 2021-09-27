# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets

import utils.mc_dropout
import utils.metrics
import utils.model

# %%
device = "cpu"

# %%
data_train = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=True, transform=torchvision.transforms.ToTensor())
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=32,
                                                shuffle=True,
                                                )

data_test = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=False, transform=torchvision.transforms.ToTensor())
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=32,
                                               shuffle=False)


# %%
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %%


class NeuralNetwork(nn.Module):
    def __init__(self, p_dropout):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %%
model = NeuralNetwork(p_dropout=0).to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train_progress = utils.model.train_model(
    model, 5, optimizer, criterion, data_loaders, dataset_sizes)


# %%
def plot_uncertainties(progress):
    correct = progress.predictions == progress.labels
    mc_dropout_confidence = 1 - val_progress.dropout_variances / \
        val_progress.dropout_variances.max()
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


# %%
utils.mc_dropout.set_dropout_p(model, model, .5)
val_progress = utils.model.run_validation(model, data_loaders["val"])

# %%
plot_uncertainties(val_progress)

# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
for label, idx in [("MC dropout", np.argsort(val_progress.dropout_variances)[::-1]),
                   ("Confidence", np.argsort(val_progress.confidences)),
                   ("Max prob", np.argsort(val_progress.max_probs))]:
    labels = val_progress.labels[idx]
    predictions = val_progress.dropout_predictions[
        idx] if label == "MC dropout" else val_progress.predictions[idx]
    accs = utils.metrics.roc_stat(labels, predictions, step=10)
    ax.plot(np.linspace(0, 100, len(accs)), accs, label=label)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
