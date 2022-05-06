# This script trains a DeepLabV3 model with ResNet-50 backbone for segmentation on the PASCAL-VOC segmentation dataset.


from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.metrics
import utils.mc_dropout
import utils.model
import utils.visualisations


# set the name of the experiment (used to save checkpoints and log data with tensorboard)
EXPERIMENT_NAME = "voc_segmentation_deeplab_resnet50-test"

# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



model = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=True).to(device)


# Uncomment to add dropout layers
# utils.mc_dropout.add_dropout(model.backbone.layer3, model.backbone.layer3, 0.2)
# utils.mc_dropout.add_dropout(model.backbone.layer4, model.backbone.layer4, 0.2)


print(model)


class VOCTransform(object):
    """Convert the area close to object boundaries to background. It is a special class by default."""

    def __call__(self, labels):
        labels = (labels * 255).squeeze(0).to(torch.long)
        labels[labels == 255] = 0
        return labels


transforms_normalized = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    VOCTransform()
])

# use batchsize of 32 but accumulate grad through 64 samples. Basically like using batchsize of 64 but using less memory.
effective_batchsize = 64
batchsize = 32


data_train = torchvision.datasets.VOCSegmentation(
    root="VOC",
    download=True, #set this to false after the first run
    image_set="train", 
    transform=transforms_normalized, 
    target_transform=target_transforms)

train_set_size = int(len(data_train) * 0.9)
valid_set_size = len(data_train) - train_set_size
data_train, data_val = torch.utils.data.random_split(
    data_train, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(0))
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=batchsize,
                                                shuffle=False)


data_test = torchvision.datasets.VOCSegmentation(
    root="VOC", download=True, image_set="val", transform=transforms_normalized, target_transform=target_transforms)
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=batchsize,
                                               shuffle=False)

dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}


writer = SummaryWriter(comment=EXPERIMENT_NAME)


def train_model(model, num_epochs, optimizer, criterion, data_loaders, device, save_model_filename=None):
    softmax = nn.Softmax(dim=1)
    model.to(device)
    k = effective_batchsize // batchsize
    min_val_loss = 10000
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
                if isinstance(outputs, OrderedDict):
                    outputs = outputs["out"]
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    if (i+1) % k == 0 or (i+1) == len(progress_bar):
                        optimizer.step()
                        optimizer.zero_grad()

                probs = softmax(outputs).detach()
                running_corrects += torch.sum(preds == labels.data)
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])

                losses.append(loss.item())
                iou = utils.metrics.iou(preds, labels, 21).item()
                ious.append(iou)
                epoch_iou = np.mean(ious)
                epoch_loss = np.mean(losses)
                epoch_acc = running_corrects.double() / numel
                epoch_avg_max = running_maxes / numel
                progress_str = f'{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f} IOU: {epoch_iou:.2f} Avg. max. prob: {epoch_avg_max:.2f}'
                progress_bar.set_description(progress_str)
            writer.add_scalar(f"Acc/{phase}", epoch_acc, epoch)
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"IOU/{phase}", epoch_iou, epoch)

            if phase == "val" and epoch_loss < min_val_loss and save_model_filename is not None:
                min_val_loss = epoch_loss
                torch.save(model, save_model_filename)
                print(
                    f"Checkpoint with val_loss = {epoch_loss:.2f} saved.")


optimizer = torch.optim.Adam(model.classifier.parameters())
criterion = nn.CrossEntropyLoss().to(device)
train_progress = train_model(
    model, 20, optimizer, criterion, data_loaders, device, f"checkpoints/{EXPERIMENT_NAME}.pt")
