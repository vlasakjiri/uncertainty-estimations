# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from matplotlib import pyplot as plt
import utils.visualisations
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets

import utils.metrics
import utils.model

import models.mnist

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# %%
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()])

data_train = torchvision.datasets.FashionMNIST(
    "fashionmnist", download=True, train=True, transform=transforms)
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=32,
                                                shuffle=True,
                                                )

data_test = torchvision.datasets.FashionMNIST(
    "fashionmnist", download=True, train=False, transform=transforms)
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=32,
                                               shuffle=False)


# %%
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}


# %%
model = models.mnist.LeNet5_dropout(p_dropout=0).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
train_progress = utils.model.train_model(
    model, 20, optimizer, criterion, data_loaders, device)


# %%
# utils.mc_dropout.set_dropout_p(model, model, .15)
progress = utils.model.run_validation(
    model, data_loaders["val"], utils.metrics.Progress(), device, use_mc_dropout=True)


# %%
dropout_max_probs = progress.dropout_outputs.max(axis=-1)

utils.visualisations.samples_removed_vs_acc([
    ("Max prob", np.argsort(progress.max_probs)),
    ("Dropout max probs", np.argsort(dropout_max_probs))],
    progress.labels,
    progress.predictions,
    progress.dropout_predictions)


counts = utils.visualisations.calibration_graph([
    ("Max prob", progress.max_probs, np.argsort(progress.max_probs)),
    ("Dropout max probs", dropout_max_probs, np.argsort(dropout_max_probs))],
    progress.labels,
    progress.predictions,
    progress.dropout_predictions)


# %%
plt.bar(np.linspace(0.05, 0.95, 10), counts[0], 0.1)

# %%
