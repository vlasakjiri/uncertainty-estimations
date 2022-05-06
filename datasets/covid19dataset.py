# Pytorch dataset for the MedSeg Covid19 dataset.

import torch
import os
import cv2


class Covid19Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, multi=True, transform=None, target_transform=None):
        image_folder = os.path.join(root_dir, "im", "256")
        self.multi = multi
        gt_folder = os.path.join(
            root_dir, "mask", "multi" if self.multi else "binary", "256")
        self.codes = [name.split(".")[0][1:]
                      for name in os.listdir(image_folder)]
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

        img = list(filter(lambda name: code in name, self.image_names))[0]
        gt = list(filter(lambda name: code in name, self.gt_names))[0]

        img = cv2.imread(img, 0) / 255
        gt = cv2.imread(gt, 0) / 255

        max_class = 3 if self.multi else 1
        gt = gt * max_class

        img = torch.as_tensor(img, dtype=torch.float)
        gt = torch.as_tensor(gt, dtype=torch.long).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            gt = self.target_transform(gt)

        return img.unsqueeze(0), gt.squeeze(0)
