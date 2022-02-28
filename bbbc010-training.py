# %%
import torchmetrics
import torch.nn.functional as F
from typing import OrderedDict
from tqdm import tqdm
import cv2
from PIL import Image, ImageOps
import os
import models.unet_model
import models.resnet
import models.resnet_dropout
import utils.model
import utils.metrics
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch
import numpy as np
import utils.visualisations
from matplotlib import pyplot as plt
from utils.temperature_scaling import ModelWithTemperature
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter


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


class BBBC010Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        image_folder = os.path.join(root_dir, "BBBC010_v2_images")
        gt_folder = os.path.join(root_dir, "BBBC010_v1_foreground")
        self.codes = [name.split("_")[0] for name in os.listdir(gt_folder)]
        self.image_names = [os.path.join(image_folder, name)
                            for name in os.listdir(image_folder)]
        self.gt_names = [os.path.join(gt_folder, name)
                         for name in os.listdir(gt_folder)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        img1 = list(filter(lambda name: code +
                    "_w1" in name, self.image_names))[0]
        img2 = list(filter(lambda name: code +
                    "_w2" in name, self.image_names))[0]
        gt = list(filter(lambda name: code in name, self.gt_names))[0]

        img1 = cv2.imread(img1, 0)
        img2 = cv2.imread(img2, 0)
        gt = cv2.imread(gt, 0)

        img = np.zeros((2, *img1.shape))

        img[0] = img1 / img1.max()
        img[1] = img2 / img2.max()
        gt = gt / 255
        gt[gt < 0.5] = 0
        gt[gt >= 0.5] = 1

        img = torch.as_tensor(img, dtype=torch.float)
        gt = torch.as_tensor(gt, dtype=torch.long).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            gt = self.target_transform(gt)

        return img, gt.squeeze(0)


# %%
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])

target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])

data = BBBC010Dataset("BBBC010", transform=transform,
                      target_transform=target_transform)


data_train, data_test = torch.utils.data.random_split(
    data, [60, 40], generator=torch.Generator().manual_seed(0))

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
model = models.unet_model.UNet(2, 2)
model

# %%

writer = SummaryWriter(comment="unet_BBBC010")

effective_batchsize = 16
batchsize = 16


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


optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss().to(device)
train_progress = train_model(
    model, 100, optimizer, criterion, data_loaders, device, "unet_BBBC010.pt")

# %%
# scaled_model = ModelWithTemperature(model)
# scaled_model.temperature = torch.nn.Parameter(torch.tensor(1.458))
# # scaled_model.set_temperature(data_loader, device)

# # %%


# def run_validation(model, data_loader, test_progress: utils.metrics.Progress, device, mc_dropout_iters=0):
#     softmax = nn.Softmax(dim=1)
#     progress_bar = tqdm(data_loader)
#     count = 0
#     running_corrects = 0
#     model = model.to(device)
#     softmax = torch.nn.Softmax(dim=1)
#     for inputs, labels in progress_bar:
#         # labels = (labels * 255).squeeze().to(torch.uint8)
#         # labels[labels == 255] = 21
#         inputs = inputs.to(device)
#         count += labels.numel()
#         model.eval()
#         with torch.no_grad():
#             logits = model(inputs)
#         if isinstance(logits, OrderedDict):
#             logits = logits["out"]
#         logits = logits.cpu()
#         probs = softmax(logits)
#         max_probs, preds = torch.max(probs, 1)
#         # print(iou(preds, labels))
#         # return
#         running_corrects += np.count_nonzero(preds == labels.squeeze(1))
#         # print(torchmetrics.functional.iou(
#         # preds, labels, ignore_index=21, num_classes=22))
#         if mc_dropout_iters > 0:
#             mc_means, mc_vars = utils.mc_dropout.mc_dropout(
#                 model, inputs, logits.shape[1:], T=mc_dropout_iters)
#             # batch_nll = - utils.mc_dropout.compute_log_likelihood(
#             #     mc_means, torch.nn.functional.one_hot(labels, num_classes=mc_means.shape[-1]), torch.sqrt(mc_vars))
#             batch_nll = torch.tensor([0])
#             mc_predictions = mc_means.argmax(axis=1)
#             test_progress.dropout_outputs.append(mc_means.numpy())
#             test_progress.dropout_predictions = np.append(
#                 test_progress.dropout_predictions, mc_predictions)
#             test_progress.dropout_variances = np.append(
#                 test_progress.dropout_variances, mc_vars)

#         test_progress.update(preds, labels, probs, logits)
#         progress_bar.set_description(
#             f"Avg. acc.: {100*running_corrects/count:.2f}")

#     test_progress.predictions = np.concatenate(test_progress.predictions)
#     test_progress.logits = np.concatenate(test_progress.logits)
#     test_progress.probs = np.concatenate(test_progress.probs)
#     if mc_dropout_iters > 0:
#         test_progress.dropout_outputs = np.concatenate(
#             test_progress.dropout_outputs)
#     return test_progress


# progress = run_validation(
#     model, data_loader_test, utils.metrics.Progress(), device, mc_dropout_iters=0)

# # progress_scaled = run_validation(
# #     scaled_model, data_loader, utils.metrics.Progress(), device, mc_dropout_iters=0)

# # %%
# progress.predictions.size, progress.labels.size

# # %%


# print(torchmetrics.functional.jaccard_index(preds[:64], labels[:64]))


# # print(torchmetrics.functional.jaccard_index(preds_mcd, labels))


# # %%
# def iou(preds, labels):
#     preds = F.one_hot(preds)
#     labels = F.one_hot(labels)
#     intersection = (preds & labels).sum((0, 1, 2))
#     union = (preds | labels).sum((0, 1, 2))
#     iou = (intersection) / (union)
#     print(iou)

#     return iou.nanmean()


# preds = torch.Tensor(progress.predictions).to(torch.long)

# # preds_mcd = torch.Tensor(progress.dropout_predictions.reshape(progress.predictions.shape)).to(torch.uint8)


# labels = torch.Tensor(progress.labels.reshape(
#     progress.predictions.shape)).to(torch.long)

# print(iou(preds, labels))
# # torchmetrics.functional.jaccard_index(preds[:32], labels[:32],absent_score=torch.nan, reduction="none").nanmean(), torchmetrics.functional.jaccard_index(preds[32:64], labels[32:64], absent_score=torch.nan, reduction="none").nanmean(), torchmetrics.functional.jaccard_index(preds[:64], labels[:64], absent_score=torch.nan, reduction="none").nanmean()


# # %%
# (0.5701 + 0.4562) / 2

# # %%
# preds.shape, labels.shape

# # %%
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(decode_segmap(progress.predictions[2]))
# axs[1].imshow(decode_segmap(labels[2]))

# # %%
# # nll = nn.CrossEntropyLoss()(torch.tensor(progress.logits), torch.tensor(
# #     progress.labels, dtype=torch.long)).item()
# # print(
# #     f"Accuracy: {(progress.predictions==progress.labels).sum()*100/len(progress.labels):.2f}%, "
# #     f"NLL: {nll:4f}"
# # )

# # # mc_logits = progress.dropout_logits.mean(axis=0)
# # dropout_max_probs = progress.dropout_outputs.max(axis=-1)

# # utils.visualisations.samples_removed_vs_acc([
# #     ("Max prob", np.argsort(progress.max_probs)),
# #     ("Dropout max probs", np.argsort(dropout_max_probs))],
# #     progress.labels,
# #     progress.predictions,
# #     progress.dropout_predictions)

# dropout_max_probs = progress.dropout_outputs.max(axis=1).ravel()

# utils.visualisations.calibration_graph([
#     ("Baseline", progress.max_probs, np.argsort(
#         progress.max_probs), progress.predictions.ravel()),
#     ("MCD", dropout_max_probs, np.argsort(dropout_max_probs),
#      progress.dropout_outputs.argmax(axis=1).ravel()),
#     # ("Temp scaling", progress_scaled.max_probs, np.argsort(progress_scaled.max_probs), progress_scaled.predictions.ravel()),


# ],
#     progress.labels,
# )

# # %%
# curves = []

# correct = progress.predictions.ravel() == progress.labels.ravel()
# fpr, tpr, _ = metrics.roc_curve(correct, progress.max_probs)
# roc_auc = metrics.auc(fpr, tpr)

# prec, recall, _ = metrics.precision_recall_curve(correct, progress.max_probs)
# aupr = metrics.auc(recall, prec)
# curves.append({
#     "fpr": fpr,
#     "tpr": tpr,
#     "auroc": roc_auc,
#     "prec": prec,
#     "recall": recall,
#     "aupr": aupr,
#     "label": "Softmax"
# })

# # correct = progress.dropout_predictions == progress.labels
# # dropout_max_probs = progress.dropout_outputs.max(axis=-1)
# # fpr, tpr, _ = metrics.roc_curve(correct, dropout_max_probs)
# # roc_auc = metrics.auc(fpr, tpr)

# # prec, recall, _ = metrics.precision_recall_curve(correct, dropout_max_probs)
# # aupr = metrics.auc(recall, prec)

# # curves.append({
# #     "fpr": fpr,
# #     "tpr": tpr,
# #     "auroc": roc_auc,
# #     "prec": prec,
# #     "recall": recall,
# #     "aupr": aupr,
# #     "label": "MC Dropout"
# # })


# plt.figure(figsize=(14, 8))
# plt.title('Receiver Operating Characteristic')
# for curve in curves:
#     plt.plot(curve["fpr"], curve["tpr"],
#              label=f"{curve['label']}: AUC = {curve['auroc']:.3f}")
# plt.plot([0, 1], [0, 1], 'k--', label=f"No skill: AUC = 0.5")
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.legend()

# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# # %%
# no_skill = correct.sum() / len(correct)
# plt.figure(figsize=(14, 8))
# plt.title('Precision Recall')
# for curve in curves:
#     plt.plot(curve["recall"], curve["prec"],
#              label=f"{curve['label']}: AUC = {curve['aupr']:.3f}")
# # axis labels
# plt.plot([0, 1], [no_skill, no_skill], color="k",
#          linestyle='--', label=f'No skill: AUC = {no_skill:.3f}')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# # show the legend
# plt.legend()
# # show the plot
# plt.show()

# # %%
