# This script trains LeNet-5 model for classification on the FMNIST dataset.

# %%
import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

import models.lenet
import utils.model

# choose the name of the experiment (used to save checkpoints and log data with tensorboard)
EXPERIMENT_NAME = "lenet-fmnist"

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# uncomment the model architecture that you want to use (LeNet5 or LeNet5 with dropout)
model = models.lenet.LeNet5(n_channels=1).to(device)
# model = models.lenet.LeNet5_dropout(p_dropout=0.2).to(device)


# %%
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
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

print(model)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

utils.model.train_model(
    model, 20, optimizer, criterion, data_loaders, device, f"checkpoints/{EXPERIMENT_NAME}.pt", SummaryWriter(comment=EXPERIMENT_NAME))
