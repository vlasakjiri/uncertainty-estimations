# %%
import torchmetrics
import torch.nn.functional as F
from tqdm import tqdm
import os
import models.unet_model
import utils.model
import utils.metrics
import torchvision.datasets
import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils.temperature_scaling import ModelWithTemperature
from datasets.covid19dataset import Covid19Dataset
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter


EXPERIMENT_NAME = "unet_dropout_covid19"

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
transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(256),
    # torchvision.transforms.CenterCrop(224),
])

target_transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(256),
    # torchvision.transforms.CenterCrop(224),
])

data = Covid19Dataset("covid19", multi=True, transform=transform,
                      target_transform=target_transform)


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

# model = torch.load("checkpoints/VOC_segmentation_deeplabv3_mobilenet_v3_large.pt")
# model = torch.load("checkpoints/unet_unewighted.pt")


# %%

writer = SummaryWriter(comment=EXPERIMENT_NAME)

effective_batchsize = 10
batchsize = 10


def train_model(model, num_epochs, optimizer, criterion, data_loaders, device, save_model_filename=None):
    softmax = nn.Softmax(dim=1)
    model.to(device)
    k = effective_batchsize // batchsize
    max_iou = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 10, flush=True)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_corrects = 0
            running_maxes = 0.0
            losses = []
            ious = []
            numel = 0
            progress_bar = tqdm(data_loaders[phase])
            optimizer.zero_grad()
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                numel += labels.numel()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    if (i+1) % k == 0 or (i+1) == len(progress_bar):
                        optimizer.step()
                        optimizer.zero_grad()

                probs = softmax(outputs).detach()
                # statistics
                running_corrects += torch.sum(preds == labels.data)
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])

                losses.append(loss.item())
                iou = utils.metrics.iou(preds, labels, 4).item()
                ious.append(iou)
                epoch_iou = np.mean(ious)
                epoch_loss = np.mean(losses)
                epoch_acc = running_corrects.double() / numel
                # epoch_entropy = running_entropy / count
                epoch_avg_max = running_maxes / numel
                progress_str = f'{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f} IOU: {epoch_iou:.2f} Avg. max. prob: {epoch_avg_max:.2f}'
                progress_bar.set_description(progress_str)
            writer.add_scalar(f"Acc/{phase}", epoch_acc, epoch)
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"IOU/{phase}", epoch_iou, epoch)

            # iou = torchmetrics.functional.jaccard_index(preds, labels).item()
            # print(iou)
            # print(loss)
            if phase == "val" and epoch_iou > max_iou and save_model_filename is not None:
                max_iou = epoch_iou
                torch.save(model, save_model_filename)
                print(
                    f"Checkpoint with val IOU = {epoch_iou:.3f} saved.")


for i in range(5):
    model = models.unet_model.UNet(1, 4)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    weights = torch.tensor(
        utils.model.compute_segmentation_loss_weights(data_train, 4), dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weights).to(device)
    # criterion = IoULoss(weights).to(device)
    train_progress = train_model(
        model, 200, optimizer, criterion, data_loaders, device, f"checkpoints/{EXPERIMENT_NAME}-{i}.pt")
