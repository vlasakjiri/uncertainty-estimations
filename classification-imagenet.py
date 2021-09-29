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

# %%
device = "cpu"

# %%
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
])

transforms_normalized = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = ImageNetV2Dataset(
    transform=transforms, location="imagenet")
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=256,
                                          shuffle=False)

data_normalized = ImageNetV2Dataset(
    transform=transforms_normalized, location="imagenet")
data_loader_normalized = torch.utils.data.DataLoader(data_normalized,
                                                     batch_size=256,
                                                     shuffle=False)


# %%
class_names = pickle.load(urllib.request.urlopen(
    'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))


# %%
mobilenet_small = torchvision.models.mobilenet.mobilenet_v3_small(
    pretrained=True)

# %%
progress_normalized = utils.model.run_validation(
    mobilenet_small, data_loader_normalized, utils.metrics.Progress())
progress_normalized.probs = np.concatenate(progress_normalized.probs)


# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
for label, idx in [("Confidence", np.argsort(progress_normalized.confidences)),
                   ("Max prob", np.argsort(progress_normalized.max_probs))]:
    labels = progress_normalized.labels[idx]
    top5preds = progress_normalized.probs.argsort(axis=1)[:, :-6:-1][idx]
    accs = []
    correct = [label in preds for label, preds in zip(
        labels, top5preds)]
    for _ in range(len(labels)//10):
        accs.append(sum(correct)/len(correct))
        correct = correct[10:]
    ax.plot(np.linspace(0, 100, len(accs)), accs, label=label)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.set_title("Top 5 accuracy when removing most uncertain samples")
ax.legend()


# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
for label, idx in [("MC dropout", np.argsort(progress_normalized.dropout_variances)[::-1]),
                   ("Confidence", np.argsort(progress_normalized.confidences)),
                   ("Max prob", np.argsort(progress_normalized.max_probs))]:
    labels = progress_normalized.labels[idx]
    predictions = progress_normalized.dropout_predictions[
        idx] if label == "MC dropout" else progress_normalized.predictions[idx]
    accs = utils.metrics.roc_stat(labels, predictions, step=10)
    ax.plot(np.linspace(0, 100, len(accs)), accs, label=label)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.set_title("Top 1 accuracy when removing most uncertain samples")
ax.legend()


# %%
inputs, classes = next(iter(data_loader))

fig, axs = plt.subplots(1, 5, figsize=(15, 5))
softmax = torch.nn.Softmax(dim=1)
model = mobilenet_small
model.eval()
for i in range(5):
    x = inputs[i]
    y = classes[i]
    axs[i].axis("off")
    with torch.no_grad():
        outputs = model(torch.unsqueeze(x, 0))
    probs = softmax(outputs)
    _, preds = torch.max(outputs, 1)
    confidence = 1-utils.metrics.normalized_entropy(probs, axis=1)
    max_prob = torch.max(probs, dim=1)
    axs[i].set_title(f'''Confidence: {confidence[0]:.4f}
    Max prob: {max_prob[0].item():.4f}
    Predicted: {class_names[max_prob[1].item()]}
    ''')
    axs[i].imshow(x.permute(1, 2, 0))
    # x[0] = torch.tensor(rotate(x[0], -10))

# %%
