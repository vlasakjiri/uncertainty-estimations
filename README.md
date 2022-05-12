# Assessment of Uncertainty of Neural Net Predictions in the Tasks of Classification, Detection and Segmentation

This repository provides the code used in my bachelor`s thesis "Assessment of Uncertainty of Neural Net Predictions in the Tasks of Classification, Detection and Segmentation".

You can watch the presentational video: https://youtu.be/rt9T6uYYrIQ

## Installation
with conda:
```
conda env create --file environment.yml
```
with pip:
```
pip install -r requirements.txt
```
## Repository structure
The training scripts and evaluating jupyter notebooks are located in the top level directory. They are named by the task (classification, segmentation, detection) and dataset used.

Code for evaluation, training and uncertainty estimation methods is located in the ``utils`` folder.

Code for the models is located in the ``models`` folder.

Code for data transformations and pytorch datasets is located in the ``datasets`` directory.

The ``figures`` folder contains all of the figures used in the paper.

The ``experiments`` folder contains exported results of evaluation on shifted datasets.

## Pre-trained models
Download the pre-trained models from https://drive.google.com/drive/folders/1Uzw0pO-NPe6l5SGLFjZEqQRggZMfMFAl?usp=sharing
and place them in the ``checkpoints`` folder.


## Training
The training scripts are .py files named by the dataset used. 

Example: training U-Net on the MedSeg Covid19 dataset:
```
python segmentation-covid19-training.py
```
You can change the model architecture used by uncommenting lines in the file. For example:

```python
# standard model
model = models.unet_model.UNet(1, 4)

# dropout model
# model = models.unet_model.UNet_Dropout(1, 4, p=0.1)
```

## Evaluation
To evaluate the models, use the jupyter notebooks.
