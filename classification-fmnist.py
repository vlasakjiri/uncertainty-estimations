# %%
from torchensemble import VotingClassifier
import pickle
import models.mnist
import utils.model
import utils.metrics
import torchvision.datasets
import torch.nn as nn
import torch
import numpy as np
import utils.visualisations
from matplotlib import pyplot as plt


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
                                             torchvision.transforms.RandomRotation(
                                                 degrees=(0, 0)),
                                             torchvision.transforms.ToTensor()])


data_train = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=True, transform=transforms)

train_set_size = int(len(data_train) * 0.9)
valid_set_size = len(data_train) - train_set_size
data_train, data_val = torch.utils.data.random_split(
    data_train, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(0))

data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=32,
                                                shuffle=True,
                                                )

transforms_test = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                  torchvision.transforms.RandomRotation(
                                                      degrees=(0, 0)),
                                                  torchvision.transforms.ToTensor()])

data_loader_val = torch.utils.data.DataLoader(data_val,
                                              batch_size=32,
                                              shuffle=False)

data_test = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=False, transform=transforms_test)
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=32,
                                               shuffle=False)

# %%
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}

# %%
model = models.mnist.LeNet5(n_channels=1)


ensemble = VotingClassifier(
    estimator=model,               # here is your deep learning model
    n_estimators=5,                        # number of base estimators
)
# Set the criterion
criterion = nn.CrossEntropyLoss()           # training objective
ensemble.set_criterion(criterion)

# Set the optimizer
ensemble.set_optimizer(
    "Adam",                                 # type of parameter optimizer
)


# Train the ensemble
ensemble.fit(
    data_loader_train,
    epochs=20,                          # number of training epochs
    save_model=True,
    save_dir='checkpoints'
)


# model = torch.load("models/fmnist_lenet_0dropout_all")
# model_dropout = torch.load("models/fmnist_lenet_0.5dropout_all")
# model_dropout.load_state_dict(model.state_dict())
# model = model_dropout
# utils.mc_dropout.set_dropout_p(model, model, .03)
# model.feature_extractor[9] = torch.nn.Dropout2d(p=0.03)


# model_dropout = torch.load("models/fmnist_lenet_0.5dropout_all")

# %%
# model = models.mnist.LeNet5_dropout(p_dropout=0).to(device)
# print(model)
# optimizer = torch.optim.Adam(model_dropout.parameters())
# criterion = nn.CrossEntropyLoss()

# utils.model.train_model(
#     model_dropout, 20, optimizer, criterion, data_loaders, device, "checkpoints/lenet-fmnist-dropout0.2.pt")

# %%
