# %%

import random
import matplotlib.patches as patches
import pprint
from typing import OrderedDict

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

import models.resnet
import models.resnet_dropout
import models.unet_model
import utils.detection_metrics
import utils.metrics
import utils.model
import utils.visualisations
from utils.temperature_scaling import ModelWithTemperature

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
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_DICT = {cls: i for i, cls in enumerate(VOC_CLASSES)}


class VOCTransform(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, label):
        annotation = label["annotation"]
        id = int(annotation["filename"].replace(".jpg", "").replace("_", ""))
        out = {
            "boxes": [],
            "labels": [],
            "image_id": torch.tensor(id, dtype=torch.int64),
            "area": [],
            "iscrowd": []
        }
        for obj in annotation["object"]:
            box_dict = obj["bndbox"]
            box = [float(box_dict["xmin"]), float(box_dict["ymin"]),
                   float(box_dict["xmax"]), float(box_dict["ymax"])]
            out["boxes"].append(box)
            out["labels"].append(VOC_DICT[obj["name"]])
            out["area"].append((box[2]-box[0]) * (box[3] - box[1]))
            out["iscrowd"].append(0)
        out["boxes"] = torch.as_tensor(out["boxes"], dtype=torch.float)
        out["labels"] = torch.as_tensor(out["labels"], dtype=torch.int64)
        out["area"] = torch.as_tensor(out["area"], dtype=torch.float)
        out["iscrowd"] = torch.as_tensor(out["iscrowd"], dtype=torch.uint8)
        return out


def collate_fn(batch):
    return list(zip(*batch))


transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip()
])


target_transforms = torchvision.transforms.Compose([
    VOCTransform()
])


data_train = torchvision.datasets.VOCDetection(
    root="VOC", download=True, image_set="train", transform=transforms_train, target_transform=VOCTransform())
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=16,
                                                shuffle=True,
                                                collate_fn=collate_fn)


data_test = torchvision.datasets.VOCDetection(
    root="VOC", download=True, image_set="val", transform=torchvision.transforms.ToTensor(), target_transform=VOCTransform())
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=16,
                                               shuffle=False,
                                               collate_fn=collate_fn)

# %%
# model = models.unet_model.UNet(3, 21).to(device)
# model = torch.load("checkpoints/faster_rcnn_box_predictor_dropout0.2.pt")
model = torch.load("checkpoints/ssd_detection_VOC.pt")


# %%
model

# %%
model.to(device)

# %%
len(data_train)

# %%


def bbox_dict_to_tensor(pred: dict):
    if "scores" not in pred:
        pred["scores"] = torch.ones_like(pred["labels"])
    return torch.cat((pred["labels"].unsqueeze(1), pred["scores"].unsqueeze(
        1), pred["boxes"]), 1).tolist()

# %%


class Observation:
    def __init__(self, bbox):
        self.bboxes = [bbox]
        self.mean = bbox
        self.prediction = bbox[1]

    def append(self, bbox):
        self.bboxes.append(bbox)
        self.mean = np.mean(self.bboxes, axis=0).tolist()

    def should_be_grouped(self, bbox, iou_threshold=0.95):
        if int(self.prediction) != int(bbox[1]):
            return False
        if utils.detection_metrics.intersection_over_union(torch.tensor(self.mean[3:]), torch.tensor(bbox[3:])) > iou_threshold:
            return True
        return False

    def __repr__(self):
        return f"{self.mean}, {len(self.bboxes)} bboxes combined"


def combine_boxes(preds, iou_threshold):
    observations = []
    for i in preds:
        for bbox in i:
            grouped = False
            for observation in observations:
                if observation.should_be_grouped(bbox, iou_threshold):
                    observation.append(bbox)
                    grouped = True
                    break
            if not grouped:
                observations.append(Observation(bbox))
    return observations


# %%


def validate_model(model, data_loader, device):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []

    train_idx = 0
    progress_bar = tqdm(data_loader)
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        with torch.no_grad():
            out = model(images)
        for idx, pred in enumerate(out):
            bboxes = bbox_dict_to_tensor(pred)
            true_bboxes = bbox_dict_to_tensor(targets[idx])
            nms_boxes = bboxes
            # nms_boxes = utils.detection_metrics.nms(
            #     bboxes,
            #     iou_threshold=0.95,
            #     threshold=0.5,
            # )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes:
                # many will get converted to 0 pred
                # if box[1] > threshold:
                all_true_boxes.append([train_idx] + box)
            train_idx += 1
    AP_50 = utils.detection_metrics.mean_average_precision(
        all_pred_boxes, all_true_boxes, iou_threshold=0.5, num_classes=21)
    AP_75 = utils.detection_metrics.mean_average_precision(
        all_pred_boxes, all_true_boxes, iou_threshold=0.75, num_classes=21)
    print(f'AP 50: {AP_50:.2f} AP 75: {AP_75:.2f}')
    return all_pred_boxes, all_true_boxes


def validate_model_mcd(model, data_loader, device, mcd_iters=10):
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0
    stable_train_idx = 0
    progress_bar = tqdm(data_loader)
    model.eval()
    utils.mc_dropout.set_training_mode_for_dropout(model, True)
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        batch_preds = [[[] for j in range(mcd_iters)]
                       for _ in range(len(images))]

        for idx, target in enumerate(targets):
            true_bboxes = bbox_dict_to_tensor(target)
            for box in true_bboxes:
                all_true_boxes.append([stable_train_idx + idx] + box)

        for i in range(mcd_iters):
            train_idx = stable_train_idx
            with torch.no_grad():
                out = model(images)

            for idx, pred in enumerate(out):
                bboxes = bbox_dict_to_tensor(pred)
                true_bboxes = bbox_dict_to_tensor(targets[idx])
                nms_boxes = utils.detection_metrics.nms(
                    bboxes,
                    iou_threshold=0.95,
                    threshold=0.5,
                )

                for nms_box in nms_boxes:
                    batch_preds[idx][i].append([train_idx] + nms_box)

                train_idx += 1
        stable_train_idx = train_idx
        for pred in batch_preds:
            observations = combine_boxes(pred, iou_threshold=0.9)
            combined_boxes = [box.mean for box in observations if box.mean[2]
                              > .0 and len(box.bboxes) >= 2]
            for j in range(len(combined_boxes)):
                combined_boxes[j][0] = int(combined_boxes[j][0])
            all_pred_boxes.extend(combined_boxes)

        # combine batch preds
    utils.mc_dropout.set_training_mode_for_dropout(model, False)
    AP_50 = utils.detection_metrics.mean_average_precision(
        all_pred_boxes, all_true_boxes, iou_threshold=0.5, num_classes=21)
    AP_75 = utils.detection_metrics.mean_average_precision(
        all_pred_boxes, all_true_boxes, iou_threshold=0.75, num_classes=21)
    print(f'AP 50: {AP_50:.2f} AP 75: {AP_75:.2f}')
    return all_pred_boxes, all_true_boxes


# pred_boxes, true_boxes = validate_model_mcd(
#     model,  data_loader_test, device, mcd_iters=10)

pred_boxes, true_boxes = validate_model(
    model,  data_loader_test, device)

# %%
type(true_boxes[1])

# %%


def plot_image(image, boxes, gt_boxes):
    """Plots predicted bounding boxes on the image"""
    if isinstance(image, torch.Tensor):
        image = torch.permute(image, (1, 2, 0))
    im = np.array(image)

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(15, 15))
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        cls = box[1]
        confidence = box[2]
        box = box[3:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"

        # upper_left_x = box[0] - box[2] / 2
        # upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=5*confidence,
            edgecolor="r",
            facecolor="none"
        )
        plt.text(box[0], box[1], VOC_CLASSES[int(cls)], c="r")
        # Add the patch to the Axes
        ax.add_patch(rect)

    for box in gt_boxes:
        cls = box[1]
        box = box[3:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"

        # upper_left_x = box[0] - box[2] / 2
        # upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="g",
            facecolor="none"

        )
        plt.text(box[0], box[1], VOC_CLASSES[int(cls)], c="g")

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

# %%


def correct_boxes(preds, gt, num_classes, iou_threshold=0.5):
    max_idx = gt[-1][0]
    TP = []
    FP = []
    FN = []

    pred_boxes_grouped = [[] for _ in range(max_idx + 1)]
    true_boxes_grouped = [[] for _ in range(max_idx + 1)]
    for box in preds:
        pred_boxes_grouped[box[0]].append(box)
    for box in gt:
        true_boxes_grouped[box[0]].append(box)

    for pred_boxes, true_boxes in zip(pred_boxes_grouped, true_boxes_grouped):
        pred_boxes.sort(key=lambda box: box[2], reverse=True)
        for pred_box_idx, pred_box in enumerate(pred_boxes[:]):
            best_iou = 0
            best_iou_idx = None
            for true_box_idx, true_box in enumerate(true_boxes):
                if true_box[1] == pred_box[1]:
                    iou = utils.detection_metrics.intersection_over_union(
                        torch.tensor(pred_box[3:]),
                        torch.tensor(true_box[3:]))
                    if iou > best_iou:
                        best_iou = iou
                        best_iou_idx = true_box_idx
            if best_iou > iou_threshold:
                TP.append(pred_box)
                true_boxes.remove(true_boxes[best_iou_idx])
                pred_boxes.remove(pred_box)

        for box in pred_boxes:
            FP.append(box)
        for box in true_boxes:
            FN.append(box)
    return TP, FP, FN


# %%
utils.mc_dropout.set_training_mode_for_dropout(model, True)
# i = random.randint(0, len(data_test))
i = 2545
preds = []
for _ in range(20):
    passes = []
    pred = model(data_test[i][0].unsqueeze(0).to(device))
    for img in pred:
        pred = bbox_dict_to_tensor(img)
        pred = utils.detection_metrics.nms(pred, 0.95, 0.5)
        passes.append(list(map(lambda box: [i] + box, pred)))
    preds.append(passes)
# preds = list(filter(lambda box: box[0]==i, pred_boxes))
# preds = utils.detection_metrics.nms(
#     preds,
#     iou_threshold=0.5,
#     threshold=0.5,
# )
gt = list(filter(lambda box: box[0] == i, true_boxes))
plot_image(data_test[i][0], preds[1], gt)

# %%
TP, FP, FN = correct_boxes(pred_boxes, true_boxes, len(VOC_CLASSES))

# %%
len(TP), len(FP), len(FN)

# %%
len(preds[0][0])


# %%
preds[0][0], preds[1][0]

# %%


def combine_boxes(preds):
    observations = []
    for i in preds:
        for bbox in i:
            grouped = False
            for observation in observations:
                if observation.should_be_grouped(bbox, iou_threshold=0.95):
                    observation.append(bbox)
                    grouped = True
                    break
            if not grouped:
                observations.append(Observation(bbox))
    return observations


combined = combine_boxes(preds)


# %%
combined_boxes = [box.mean for box in combined if box.mean[2]
                  > .0 and len(box.bboxes) >= 10]

# %%
combined

# %%
plot_image(data_test[i][0], combined_boxes, [])

# %%
utils.detection_metrics.mean_average_precision(pred_boxes, true_boxes, 0.9, 21)

# %%
res = [[] for _ in range(pred_boxes[-1][0] + 1)]
for box in pred_boxes:
    res[box[0]].append(box)


# %%
TP, FP, FN = correct_boxes(pred_boxes, true_boxes, len(VOC_CLASSES))


# %%
len(TP), len(FP), len(FN)

# %%
correct = [1]*len(TP) + [0]*len(FP)
probs = [box[2] for box in TP+FP]
curves = [utils.metrics.compute_model_stats(correct, probs, "test")]

# %%
len(TP)*2 + len(FP) + len(FN), len(pred_boxes) + len(true_boxes)

# %%
plt.figure(figsize=(14, 8))
plt.title('Receiver Operating Characteristic')
for curve in curves:
    plt.plot(curve["fpr"], curve["tpr"],
             label=f"{curve['label']}: AUC = {curve['auroc']:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label=f"No skill: AUC = 0.5")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# %%
plt.figure(figsize=(14, 8))
plt.title('Precision Recall')
for curve in curves:
    plt.plot(curve["recall"], curve["prec"],
             label=f"{curve['label']}: AUC = {curve['aupr']:.3f}")
# axis labels

plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()
