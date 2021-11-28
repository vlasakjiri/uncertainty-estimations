# %%

from matplotlib import pyplot as plt
import utils.visualisations
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms

import utils.metrics
import utils.model

import models.mnist
import models.resnet
import models.resnet_dropout

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# %%
transforms_train = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

transforms_test = torchvision.transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

data_train = torchvision.datasets.CIFAR100(
    "cifar100", download=True, train=True, transform=transforms_train)
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=64,
                                                shuffle=True,
                                                )

data_test = torchvision.datasets.CIFAR100(
    "cifar100", download=True, train=False, transform=transforms_test)
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=64,
                                               shuffle=False)

# %%
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}

# %%
model = models.resnet.ResNet18(num_classes=100).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
train_progress = utils.model.train_model(
    model, 200, optimizer, criterion, data_loaders, device)

# %%
torch.save(model, "models/cifar100_resnet18")

# %%
# utils.mc_dropout.set_dropout_p(model, model, .15)
progress = utils.model.run_validation(
    model, data_loaders["val"], utils.metrics.Progress(), device, use_mc_dropout=True)

# %%
nll = criterion(torch.tensor(progress.logits), torch.tensor(
    progress.labels, dtype=torch.long)).item()
print(
    f"Accuracy: {(progress.predictions==progress.labels).sum()*100/len(progress.labels):.2f}%, "
    f"NLL: {nll:4f}"
)

# mc_logits = progress.dropout_logits.mean(axis=0)
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
