# %%
import pickle
from functools import reduce
from utils.detection_utils import find_jaccard_overlap
import utils.mc_dropout
import matplotlib.patches as patches
from datasets.transforms import AddGaussianNoise
from datasets.voc_detection_dataset import PascalVOCDataset
from utils.temperature_scaling import ModelWithTemperature
from pprint import PrettyPrinter
from utils.detection_utils import calculate_mAP
import utils.visualisations
import utils.model
import utils.metrics
import utils.detection_metrics
import models.unet_model
import models.resnet_dropout
import models.resnet
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn
import torch
import sklearn.metrics as metrics
import numpy as np
from typing import OrderedDict
import pprint


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
# model = torchvision.models.detection.ssd300_vgg16(


#     pretrained=True, trainable_backbone_layers=0)


model = torch.load("checkpoints/ssd300.pt")
model.to(device)

model_dropout = torch.load("checkpoints/ssd300_dropout.pt")
model_dropout.to(device)


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

test_dataset = PascalVOCDataset("VOC",
                                split='test',
                                keep_difficult=True,
                                # transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor() ,AddGaussianNoise(std=0.3), torchvision.transforms.ToPILImage()])
                                )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                          collate_fn=test_dataset.collate_fn)


# %%
# pp = PrettyPrinter()


def validate(model, test_loader, mc_dropout_iters=0):
    model.to(device)
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    true_difficulties = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            if mc_dropout_iters > 0:
                utils.mc_dropout.set_training_mode_for_dropout(model, True)
                locs_list = []
                scores_list = []
                for _ in range(mc_dropout_iters):
                    predicted_locs, predicted_scores = model(images)
                    locs_list.append(predicted_locs)
                    scores_list.append(predicted_scores)
                predicted_locs = torch.stack(locs_list).mean(dim=0)
                predicted_scores = torch.stack(scores_list).mean(dim=0)
                utils.mc_dropout.set_training_mode_for_dropout(model, False)

            else:
                # Forward prop.
                predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.2, max_overlap=0.5,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

    return det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties

# %%


def classify_boxes(det_boxes, det_labels, det_scores, true_boxes, true_labels, iou_threshold=0.5):
    TP = []
    FP = []
    FN = []

    for img_idx, (bboxes, labels, scores, gt, gt_labels) in enumerate(zip(det_boxes, det_labels, det_scores, true_boxes, true_labels)):
        idx = torch.argsort(scores, descending=True)
        ious = find_jaccard_overlap(bboxes, gt)
        matched_gt_indexes = []
        matched_boxes_indexes = []
        for true_boxes_iou, box_idx in zip(ious[idx], idx):
            best_iou = 0
            best_iou_idx = None
            for true_boxes_idx, iou in enumerate(true_boxes_iou):
                if iou > best_iou and labels[box_idx] == gt_labels[true_boxes_idx] and true_boxes_idx not in matched_gt_indexes:
                    best_iou = iou
                    best_iou_idx = true_boxes_idx
            if best_iou > iou_threshold:
                TP.append((img_idx, box_idx.item()))
                matched_gt_indexes.append(best_iou_idx)
                matched_boxes_indexes.append(box_idx.item())
        for i, box in enumerate(gt):
            if i not in matched_gt_indexes:
                FN.append((img_idx, i))
        for i, box in enumerate(bboxes):
            if i not in matched_boxes_indexes:
                FP.append((img_idx, i))
    return TP, FP, FN


# %%


class detection_model_metrics:
    def __init__(self) -> None:
        self.map50 = []
        self.map75 = []
        self.auroc = []
        self.aupr = []
        self.TP = []
        self.FP = []
        self.FN = []
        self.confidences = []
        self.strengths = []

    def __repr__(self) -> str:
        return str(vars(self))

    def update(self, det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, strength):
        self.strengths.append(strength)
        self.confidences.append(torch.cat(det_scores).mean())
        APs, mAP = calculate_mAP(
            det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.5)
        self.map50.append(mAP)
        APs, mAP = calculate_mAP(
            det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.75)
        self.map75.append(mAP)
        TP, FP, FN = classify_boxes(
            det_boxes, det_labels, det_scores, true_boxes, true_labels)
        self.TP.append(len(TP))
        self.FP.append(len(FP))
        self.FN.append(len(FN))

        correct = [1]*len(TP) + [0]*len(FP)
        probs = []
        for img_idx, box_idx in TP+FP:
            probs.append(det_scores[img_idx][box_idx].item())
        metrics = utils.metrics.compute_model_stats(correct, probs, "test")
        self.auroc.append(metrics["auroc"])
        self.aupr.append(metrics["aupr"])


curves = {
    "Vanilla": detection_model_metrics(),
    "Dropout training": detection_model_metrics(),
    "MC Dropout": detection_model_metrics(),
}


for s in np.arange(0, 0.6, 0.05):
    print(s)

    test_dataset = PascalVOCDataset("VOC",
                                    split='test',
                                    keep_difficult=True,
                                    transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), AddGaussianNoise(std=s), torchvision.transforms.ToPILImage()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                              collate_fn=test_dataset.collate_fn)

    det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = validate(
        model, test_loader)
    curves["Vanilla"].update(
        det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, s)

    det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = validate(
        model_dropout, test_loader)
    curves["Dropout training"].update(
        det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, s)

    det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = validate(
        model_dropout, test_loader, mc_dropout_iters=20)
    curves["MC Dropout"].update(
        det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, s)

    del det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties


# %%
curves

# %%
with open("experiments/voc-detection-all-noise.pickle", "wb") as f:
    pickle.dump(curves, f)
