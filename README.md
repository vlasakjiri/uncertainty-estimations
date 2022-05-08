# Assessment of Uncertainty of Neural Net Predictions in the Tasks of Classification, Detection and Segmentation

This repository provides the code used in my bachelor`s thesis "Assessment of Uncertainty of Neural Net Predictions in the Tasks of Classification, Detection and Segmentation".

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

The ``checkpoints`` folder contains checkpoints of the trained models. To load them, use 
```python
torch.load("checkpoints/NAME OF THE MODEL")
```

The ``cifar100``, ``covid19`` and ``fmnist`` folders contain the respective datasets used for training and evaluation. The PASCAL-VOC dataset will be downloaded automatically when running the training or evaluation scripts for the first time.


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