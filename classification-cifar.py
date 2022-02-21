# %%

import sklearn.metrics as metrics
from utils.temperature_scaling import ModelWithTemperature
from matplotlib import pyplot as plt
import utils.visualisations
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms

import utils.metrics
import utils.model

import models.mnist
import models.resnet_dropout
import models.resnet
import models.mobilenet_v2

# %%
# setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# %%
transforms_train = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

transforms_test = torchvision.transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

data_train = torchvision.datasets.CIFAR100(
    "cifar100", download=True, train=True, transform=transforms_train)
data_train, data_val = torch.utils.data.random_split(data_train, [45000, 5000])

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
model = models.mobilenet_v2.MobileNetV2(
    num_classes=100, stem_stride=1).to(device)

# model = torch.load("models/cifar100_resnet18_train_val_split")
# model_dropout = torch.load("models/cifar100_resnet18_0.2dropout_all")
# model_dropout.load_state_dict(model.state_dict())
# model = model_dropout

# %%

# model = models.resnet.ResNet18(
#     num_classes=100).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
train_progress = utils.model.train_model(
    model, 200, optimizer, criterion, data_loaders, device, "checkpoints/cifar100_mobilenetv2.pt")

# %%
torch.save(model, "models/cifar100_mobilenetv2_train_val_split")

# # %%
# # model.dropout = torch.nn.Dropout(p=0)

# # %%
# scaled_model = ModelWithTemperature(model)
# scaled_model.set_temperature(data_loader_val)

# # %%
# # utils.mc_dropout.set_dropout_p(model, model, .03)
# progress = utils.model.run_validation(
#     scaled_model, data_loaders["test"], utils.metrics.Progress(), device, use_mc_dropout=True)

# # %%
# nll = nn.CrossEntropyLoss()(torch.tensor(progress.logits), torch.tensor(
#     progress.labels, dtype=torch.long)).item()
# print(
#     f"Accuracy: {(progress.predictions==progress.labels).sum()*100/len(progress.labels):.2f}%, "
#     f"NLL: {nll:4f}"
# )

# # mc_logits = progress.dropout_logits.mean(axis=0)
# dropout_max_probs = progress.dropout_outputs.max(axis=-1)

# utils.visualisations.samples_removed_vs_acc([
#     ("Max prob", np.argsort(progress.max_probs)),
#     ("Dropout max probs", np.argsort(dropout_max_probs))],
#     progress.labels,
#     progress.predictions,
#     progress.dropout_predictions)


# counts = utils.visualisations.calibration_graph([
#     ("Max prob", progress.max_probs, np.argsort(progress.max_probs)),
#     ("Dropout max probs", dropout_max_probs, np.argsort(dropout_max_probs))],
#     progress.labels,
#     progress.predictions,
#     progress.dropout_predictions)

# # %%
# curves = []

# correct = progress.predictions == progress.labels
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

# correct = progress.dropout_predictions == progress.labels
# dropout_max_probs = progress.dropout_outputs.max(axis=-1)
# fpr, tpr, _ = metrics.roc_curve(correct, dropout_max_probs)
# roc_auc = metrics.auc(fpr, tpr)

# prec, recall, _ = metrics.precision_recall_curve(correct, dropout_max_probs)
# aupr = metrics.auc(recall, prec)

# curves.append({
#     "fpr": fpr,
#     "tpr": tpr,
#     "auroc": roc_auc,
#     "prec": prec,
#     "recall": recall,
#     "aupr": aupr,
#     "label": "MC Dropout"
# })


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
