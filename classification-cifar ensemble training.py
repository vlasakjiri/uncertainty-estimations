# This script trains an ensemble model for classification on the CIFAR-100 dataset.

# %%

import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torchensemble import VotingClassifier

import models.mobilenet_v2
import models.resnet
import models.resnet_dropout

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print('Using device:', device)

# %%
# uncomment the model that you want to use (mobilenetv2 or resnet18)

model = models.mobilenet_v2.MobileNetV2(
    num_classes=100, stem_stride=1).to(device)
# model = models.resnet.ResNet18(num_classes=100).to(device)

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
ensemble = VotingClassifier(
    estimator=model,               # here is your deep learning model
    n_estimators=5,                        # number of base estimators
    cuda=device.type=="cuda"
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
    epochs=50,                          # number of training epochs
    save_model=True,
    save_dir='checkpoints'
)
