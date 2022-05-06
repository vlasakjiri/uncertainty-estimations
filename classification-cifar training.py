# This script trains resnet-18 or mobilenetv2 model for classification on the CIFAR-100 dataset.


# %%

import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import models.mobilenet_v2
import models.resnet
import models.resnet_dropout
import utils.model
import utils.visualisations

# choose the name of the experiment (used to save checkpoints and log data with tensorboard)
EXPERIMENT_NAME = "cifar100_resnet18"

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
# uncomment the model architecture that you want to use (mobilenetv2 or resnet18)

model = models.resnet.ResNet18(num_classes=100).to(device)
# model = models.resnet_dropout.ResNet18Dropout(100, p=0.1).to(device)
# model = models.mobilenet_v2.MobileNetV2(
#     num_classes=100, stem_stride=1).to(device)
# model = models.mobilenet_v2.MobileNetV2Dropout(
#     num_classes=100, stem_stride=1, p_dropout=0.1).to(device)
print(model)

# %%
transforms_train = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

transforms_test = torchvision.transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

data_train = torchvision.datasets.CIFAR100(
    "cifar100", download=True, train=True, transform=transforms_train)
data_train, data_val = torch.utils.data.random_split(
    data_train, [45000, 5000], generator=torch.Generator().manual_seed(0))


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
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train,
                "val": data_loader_test}


# %%
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
train_progress = utils.model.train_model(
    model, 200, optimizer, criterion, data_loaders, device, f"checkpoints/{EXPERIMENT_NAME}.pt", SummaryWriter(comment=EXPERIMENT_NAME))
