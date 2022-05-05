# %%
import torch
import torch.nn as nn
import torchvision.datasets
from torchensemble import VotingClassifier

import models.lenet

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
# set the model architecture
model = models.mnist.LeNet5(n_channels=1)

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
