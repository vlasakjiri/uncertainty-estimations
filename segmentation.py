# %%
import pickle
import urllib
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib

import numpy as np
import torch
import torchvision
# from torchmetrics import IoU
import torchmetrics
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

target_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])


dataset = torchvision.datasets.VOCSegmentation(
    root="VOC", download=True, image_set="val", transform=transforms_normalized, target_transform=target_transforms)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=64,
                                          shuffle=False)

# %%
VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

# %%
# # Define the helper function


def decode_segmap(image, nc=22):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                           128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 128, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

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
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
    pretrained=True)
# add_dropout(
#     model, model, 5e-3)
print(model)

# %%


def run_validation(model, data_loader, test_progress, device, use_mc_dropout=False):
    progress_bar = tqdm(data_loader)
    count = 0
    running_corrects = 0
    model = model.to(device)
    model.eval()
    max_probs_list = []
    preds_list = []
    gt_list = []
    entropies_list = []
    for inputs, gt in progress_bar:
        gt = (gt * 255).squeeze().to(torch.uint8)
        gt[gt == 255] = 21
        inputs = inputs.to(device)
        gt = gt.to(device)
        count += torch.prod(torch.tensor(gt.shape))
        with torch.no_grad():
            outputs = model(inputs)["out"].softmax(dim=1)
        # probs = softmax(outputs)
        max_probs, preds = torch.max(outputs, 1)
        preds = preds.to(torch.uint8)
        running_corrects += (preds == gt).sum()
        print(gt.element_size() * gt.nelement())
        print(torchmetrics.functional.iou(
            preds, gt, ignore_index=21, num_classes=22))
        max_probs_list.append(max_probs.cpu())
        preds_list.append(preds.cpu())
        gt_list.append(gt.cpu())
        entropies_list.append(
            utils.metrics.normalized_entropy(outputs.cpu(), axis=1))
        # print(torchmetrics.functional.accuracy(
        #     preds, gt, ignore_index=21, num_classes=22))

        # return preds.cpu(), gt.cpu()
        # print(iou(preds, gt))
        # print(running_corrects, count)
        # if use_mc_dropout:
        #     mc_output = mc_dropout(
        #         model, inputs).detach().cpu().numpy()
        #     mc_means = np.mean(mc_output, axis=0)
        #     mc_var = mc_output.var(axis=0).sum(axis=-1)
        #     test_progress.update_mcd(mc_means, mc_var)
        # test_progress.update(preds, labels, probs)
        progress_bar.set_description(
            f"Avg. acc.: {running_corrects/count:.2f}")
    # test_progress.probs = np.concatenate(test_progress.probs)
    # if use_mc_dropout:
    #     test_progress.dropout_outputs = np.concatenate(
    #         test_progress.dropout_outputs)
    max_probs_list = np.concatenate(max_probs_list)
    preds_list = np.concatenate(preds_list)
    gt_list = np.concatenate(gt_list)
    entropies_list = np.concatenate(entropies_list)
    return max_probs_list, preds_list, gt_list, entropies_list


# %%
max_probs, preds, gt, entropies = run_validation(
    model, data_loader, utils.metrics.Progress(), device, use_mc_dropout=False)


# with open("progresses/resnet18", "rb") as f:
#     progress = pickle.load(f)
# %%
fig, axs = plt.subplots(1, 2)
axs[0].imshow(decode_segmap(preds[2]))
axs[1].imshow(decode_segmap(gt[2]))


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
