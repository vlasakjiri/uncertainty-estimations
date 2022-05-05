# %%
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import models.unet_model
import utils.metrics
import utils.model
from datasets.covid19dataset import Covid19Dataset

# choose the name of the experiment (used to save checkpoints and log data with tensorboard)
EXPERIMENT_NAME = "unet_dropout_covid19"

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# %%

data = Covid19Dataset("covid19", multi=True)


data_train, data_val, data_test = torch.utils.data.random_split(
    data, [60, 10, 30], generator=torch.Generator().manual_seed(0))

data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=16,
                                                shuffle=True)


data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=16,
                                               shuffle=False)

dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}


# %%

writer = SummaryWriter(comment=EXPERIMENT_NAME)

for i in range(5):
    model = models.unet_model.UNet(1, 4)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    weights = torch.tensor(
        utils.model.compute_segmentation_loss_weights(data_train, 4), dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weights).to(device)
    train_progress = utils.model.train_model(
        model, 200, optimizer, criterion, data_loaders, device, f"checkpoints/{EXPERIMENT_NAME}-{i}.pt")
