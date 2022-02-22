# %%
import pickle
import urllib
from matplotlib import pyplot as plt
import matplotlib

import numpy as np
import torch
import torchvision
from imagenetv2_pytorch import ImageNetV2Dataset

import utils.metrics
import utils.model
import utils.mc_dropout

# %%
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# %%
transforms_normalized = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = ImageNetV2Dataset(
    transform=transforms_normalized, location="imagenet")
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=64,
                                          shuffle=False)


# %%
class_names = pickle.load(urllib.request.urlopen(
    'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))


# %%
# mobilenet_small = torchvision.models.mobilenet.mobilenet_v3_small(
#     pretrained=True)


# mobilenet_large = torchvision.models.mobilenet.mobilenet_v3_large(
#     pretrained=True)

# vgg11_bn = torchvision.models.vgg11_bn(pretrained=True, progress=False)

# %%


def add_dropout(model, block, prob, omitted_blocks=[]):
    for name, p in block.named_children():
        if p in omitted_blocks:
            continue
        if isinstance(p, torch.nn.Module):
            add_dropout(model, p, prob, omitted_blocks)
        if "relu" in p.__class__.__name__.lower():
            # bn = torch.nn.BatchNorm2d(p.num_features)
            # bn.load_state_dict(p.state_dict())
            # setattr(block, name, bn)
            sequential = torch.nn.Sequential(
                p, torch.nn.Dropout2d(p=prob, inplace=True))
            setattr(block, name, sequential)


# %%
model = torchvision.models.resnet18(pretrained=True)
# add_dropout(
#     model, model, 5e-3)
print(model)

# %%
# progress = utils.model.run_validation(
#     model, data_loader, utils.metrics.Progress(), device, use_mc_dropout=True)


with open("progresses/resnet18", "rb") as f:
    progress = pickle.load(f)

# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
for label, idx in [("MC dropout", np.argsort(progress.dropout_variances)[::-1]),
                   ("Confidence", np.argsort(progress.confidences)),
                   ("Max prob", np.argsort(progress.max_probs)),
                   ("Ideal", np.argsort(progress.predictions == progress.labels))]:
    labels = progress.labels[idx]
    predictions = progress.dropout_predictions[
        idx] if label == "MC dropout" else progress.predictions[idx]
    accs = utils.metrics.roc_stat(labels, predictions, step=10)
    x = np.linspace(0, 100, len(accs))
    ax.plot(x, accs, label=label)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.set_xlabel("Samples removed")
ax.set_ylabel("Validation accuracy")
ax.set_title("Top 1 accuracy when removing most uncertain samples")
ax.legend()


# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
for label, idx in [("MC dropout", np.argsort(progress.dropout_variances)[::-1]),
                   ("Confidence", np.argsort(progress.confidences)),
                   ("Max prob", np.argsort(progress.max_probs))]:
    labels = progress.labels[idx]
    outputs = progress.dropout_outputs if label == "MC dropout" else progress.probs
    top5preds = outputs.argsort(axis=1)[:, :-6:-1][idx]
    accs = []
    correct = [label in preds for label, preds in zip(
        labels, top5preds)]
    for _ in range(len(labels)//10):
        accs.append(sum(correct)/len(correct))
        correct = correct[10:]
    ax.plot(np.linspace(0, 100, len(accs)), accs, label=label)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.set_xlabel("Samples removed")
ax.set_ylabel("Validation accuracy")
ax.set_title("Top 5 accuracy when removing most uncertain samples")
ax.legend()

# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
bins = np.linspace(0, 1, num=10)
dropout_confidences = 1 -\
    (progress.dropout_variances / max(progress.dropout_variances))
for label, sort, idx in [("MC dropout", dropout_confidences, np.argsort(dropout_confidences)),
                         ("Confidence", progress.confidences,
                          np.argsort(progress.confidences)),
                         ("Max prob", progress.max_probs, np.argsort(progress.max_probs))]:
    sort = sort[idx]
    labels = progress.labels[idx]
    predictions = progress.predictions[idx]
    inds = np.digitize(sort, bins)
    accs = []
    for i, bin in enumerate(bins):
        idx = np.argwhere(inds == i)
        acc = (predictions[idx] == labels[idx]).sum() / \
            len(idx) if len(idx) > 0 else 0
        accs.append(acc)
    ax.plot(bins, accs, label=label)
ax.plot(bins, np.linspace(0, 1, len(bins)))
ax.set_ylim([0, 1])
ax.legend()

# %%
inputs, classes = next(iter(data_loader))
inputs = inputs.to(device)


fig, axs = plt.subplots(1, 5, figsize=(15, 5))
softmax = torch.nn.Softmax(dim=1)
model = mobilenet_large
model.eval()
for i in range(5):
    x = inputs[i]
    y = classes[i]
    axs[i].axis("off")
    with torch.no_grad():
        outputs = model(torch.unsqueeze(x, 0)).cpu()
    probs = softmax(outputs)
    _, preds = torch.max(outputs, 1)
    confidence = 1-utils.metrics.normalized_entropy(probs, axis=1)
    max_prob = torch.max(probs, dim=1)
    axs[i].set_title(f'''Confidence: {confidence[0]:.4f}
    Max prob: {max_prob[0].item():.4f}
    Predicted: {class_names[max_prob[1].item()]}
    ''')
    axs[i].imshow(x.cpu().permute(1, 2, 0))
    # x[0] = torch.tensor(rotate(x[0], -10))

# %%