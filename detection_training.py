# %%
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
from torch.utils.tensorboard import SummaryWriter
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
EXPERIMENT_NAME = "faster_rcnn_box_predictor_dropout0.5"


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

# dataset_sizes = {"train": len(data_train), "val": len(data_test)}
# data_loaders = {"train": data_loader_train, "val": data_loader_test}


# %%
# model = torchvision.models.detection.ssd300_vgg16(
#     pretrained=True, trainable_backbone_layers=0)

# model.head = torchvision.models.detection.ssd.SSDHead(
#     [512, 1024, 512, 256, 256, 256], [4, 6, 6, 6, 4, 4], 21)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True, trainable_backbone_layers=0)
num_classes = len(VOC_CLASSES)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


model.roi_heads.box_predictor.cls_score = torch.nn.Sequential(torch.nn.Dropout(p=.5),
                                                              model.roi_heads.box_predictor.cls_score)

model.roi_heads.box_predictor.bbox_pred = torch.nn.Sequential(torch.nn.Dropout(p=.5),
                                                              model.roi_heads.box_predictor.bbox_pred)

# model.roi_heads.box_head.fc6 = torch.nn.Sequential(
#     # model.roi_heads.box_head.fc6, torch.nn.Dropout(p=.2))
# model.roi_heads.box_head.fc7 = torch.nn.Sequential(
#     model.roi_heads.box_head.fc7, torch.nn.Dropout(p=.2))

# model.backbone.fpn.layer_blocks[1] = torch.nn.Sequential(
#     model.backbone.fpn.layer_blocks[1], torch.nn.Dropout2d(p=1))

model.to(device)
# model = torch.nn.parallel.DataParallel(model, device_ids=[0, 1])
# %%
print(model)

# %%
writer = SummaryWriter(comment=EXPERIMENT_NAME)


def bbox_dict_to_tensor(pred: dict):
    if "scores" not in pred:
        pred["scores"] = torch.ones_like(pred["labels"])
    return torch.cat((pred["labels"].unsqueeze(1), pred["scores"].unsqueeze(
        1), pred["boxes"]), 1).tolist()


def train_model(model, num_epochs, optimizer, data_loaders, device, save_model_filename=None):
    max_ap = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 10, flush=True)
        for phase in ["train", 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()
            progress_bar = tqdm(data_loaders[phase])
            all_pred_boxes = []
            all_true_boxes = []
            train_idx = 0
            epoch_losses = []
            for images, targets in progress_bar:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                if phase == 'train':
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        loss_dict = model(images, targets)
                        loss = sum(loss for loss in loss_dict.values())
                        loss.backward()
                        optimizer.step()
                    epoch_losses.append(loss.item())
                    progress_str = f'{phase} Loss: {np.mean(epoch_losses):.2f}'
                    progress_bar.set_description(progress_str)
                else:
                    with torch.no_grad():
                        out = model(images)
                    for idx, pred in enumerate(out):
                        bboxes = bbox_dict_to_tensor(pred)
                        true_bboxes = bbox_dict_to_tensor(targets[idx])
                        nms_boxes = utils.detection_metrics.nms(
                            bboxes,
                            iou_threshold=0.5,
                            threshold=0.5,
                            box_format="corners",
                        )

                        for nms_box in nms_boxes:
                            all_pred_boxes.append([train_idx] + nms_box)

                        for box in true_bboxes:
                            # many will get converted to 0 pred
                            # if box[1] > threshold:
                            all_true_boxes.append([train_idx] + box)
                        train_idx += 1
            if phase == "val":
                AP_50 = utils.detection_metrics.mean_average_precision(
                    all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="corners", num_classes=21)
                AP_75 = utils.detection_metrics.mean_average_precision(
                    all_pred_boxes, all_true_boxes, iou_threshold=0.75, box_format="corners", num_classes=21)
                print(f'{phase} AP 50: {AP_50:.2f} AP 75: {AP_75:.2f}')
                writer.add_scalar(f"AP 50", AP_50, epoch)
                writer.add_scalar(f"AP 75", AP_75, epoch)

                if AP_50 > max_ap and save_model_filename is not None:
                    max_ap = AP_50
                    torch.save(model, save_model_filename)
                    print(
                        f"Checkpoint with AP = {AP_50:.2f} saved.")
            else:
                writer.add_scalar(f"Train loss", np.mean(epoch_losses), epoch)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.roi_heads.box_predictor.parameters())
train_progress = train_model(
    model, 50, optimizer, {"train": data_loader_train, "val": data_loader_test}, device, f"checkpoints/{EXPERIMENT_NAME}.pt")

# %%
