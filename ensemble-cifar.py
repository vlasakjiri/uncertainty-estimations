# %%

import sklearn.metrics as metrics
from utils.temperature_scaling import ModelWithTemperature
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
import models.resnet_dropout
import models.resnet

from torchensemble import VotingClassifier

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
data_train, data_val = torch.utils.data.random_split(data_train, [45000, 5000])

data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=64,
                                                shuffle=True,
                                                )
data_loader_val = torch.utils.data.DataLoader(data_val,
                                              batch_size=64,
                                              shuffle=False,
                                              )

data_test = torchvision.datasets.CIFAR100(
    "cifar100", download=True, train=False, transform=transforms_test)
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=64,
                                               shuffle=False)

# %%
dataset_sizes = {"train": len(data_train), "val": len(
    data_val), "test": len(data_test)}
data_loaders = {"train": data_loader_train,
                "val": data_loader_val, "test": data_loader_test}


# %%
model = models.resnet.ResNet18(
    num_classes=100).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
# train_progress = utils.model.train_model(
#     model, 50, optimizer, criterion, data_loaders, device)


# %%
ensemble = VotingClassifier(estimator=model, n_estimators=5)
ensemble.set_optimizer("Adam")
ensemble.fit(train_loader=data_loader_train, epochs=50)
