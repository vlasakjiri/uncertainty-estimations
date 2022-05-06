# Pytorch dataset for the PASCAL-VOC detection dataset
# taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection 

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils.detection_utils import transform
from torchvision.datasets import VOCDetection
from utils.detection_utils import create_data_lists


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, download=False, keep_difficult=False, transforms=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        if download:
            VOCDetection(data_folder, "2012", "trainval", download=True)
            VOCDetection(data_folder, "2007", "trainval", download=True)
            VOCDetection(data_folder, "2007", "test", download=True)
            create_data_lists(os.path.join(data_folder, "VOCdevkit", "VOC2007"), os.path.join(
                data_folder, "VOCdevkit", "VOC2012"), data_folder)

        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.transforms = transforms

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        if self.transforms:
            image = self.transforms(image)

        # Apply transformations
        image, boxes, labels, difficulties = transform(
            image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        # tensor (N, 3, 300, 300), 3 lists of N tensors each
        return images, boxes, labels, difficulties
