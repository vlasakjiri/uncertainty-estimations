# %%

from typing import OrderedDict
import torchmetrics
from tqdm import tqdm
import sklearn.metrics as metrics
from utils.temperature_scaling import ModelWithTemperature
from matplotlib import pyplot as plt
import utils.visualisations
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import utils.metrics
import utils.model

import models.unet_model
import models.resnet
import models.deeplabv3

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


class VOCTransform(object):
    """Convert ndarrays in sample to Tensors."""

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
effective_batchsize = 64
batchsize = 32
data_train = torchvision.datasets.VOCSegmentation(
    root="VOC", download=True, image_set="train", transform=transforms_normalized, target_transform=target_transforms)

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


# %%
# model = torchvision.models.segmentation._deeplabv3_resnet(
#     models.resnet.ResNet18(None), 21)

model = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=True)
# model = models.unet_model.UNet(3, 21)
# utils.mc_dropout.set_dropout_p(model, model, .25)


# %%
def add_dropout(model, block, prob, omitted_blocks=[]):
    for name, p in block.named_children():
        if any(map(lambda x: isinstance(p, x), omitted_blocks)):
            continue
        if isinstance(p, torch.nn.Module) or isinstance(p, torch.nn.Sequential):
            add_dropout(model, p, prob, omitted_blocks)

        if isinstance(p, torch.nn.ReLU):
            setattr(block, name, torch.nn.Sequential(
                torch.nn.ReLU(), torch.nn.Dropout2d(p=prob)))


#             # return model
# add_dropout(model.backbone.layer3, model.backbone.layer3, 0.2)
add_dropout(model.backbone.layer4, model.backbone.layer4, 0.2)

# model.classifier[0].project[2] = torch.nn.ReLU()
# model.classifier[3] = torch.nn.ReLU()
# model.aux_classifier[2] = torch.nn.ReLU()
# backbone = torchvision.models.resnet50(
#     pretrained=False, replace_stride_with_dilation=[False, True, True])

# model = models.deeplabv3.deeplabv3_resnet(backbone, 21, False)

print(model)

# %%
model.to(device)

# %%
writer = SummaryWriter(comment="deeplab_resnet_finetune")


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
                # statistics
                running_corrects += torch.sum(preds == labels.data)
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])

                losses.append(loss.item())
                iou = utils.metrics.iou(preds, labels, 21).item()
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
            if phase == "val" and epoch_loss < min_val_loss and save_model_filename is not None:
                min_val_loss = epoch_loss
                torch.save(model, save_model_filename)
                print(
                    f"Checkpoint with val_loss = {epoch_loss:.2f} saved.")


optimizer = torch.optim.Adam(model.classifier.parameters())
weights = torch.tensor(
    utils.model.compute_segmentation_loss_weights(data_train, 21)).to(torch.float)
criterion = nn.CrossEntropyLoss().to(device)
train_progress = train_model(
    model, 20, optimizer, criterion, data_loaders, device, "checkpoints/deeplab_resnet_finetune.pt")

# torch.save(model, "models/VOC_segmentation_unet")

# %%
