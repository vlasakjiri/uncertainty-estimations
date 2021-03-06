# This file contains model related utilities (training, validation...)

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils.mc_dropout
from utils.metrics import Progress, normalized_entropy


def train_model(model, num_epochs, optimizer, criterion, data_loaders, device, save_model_filename=None, writer=None):
    """Trains a classification model"""
    softmax = nn.Softmax(dim=1)
    min_val_loss = 10000
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 10, flush=True)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            epoch_loss = 0.0
            running_corrects = 0
            running_entropy = 0.0
            running_maxes = 0.0
            count = 0
            # progress_bar = data_loaders[phase]
            progress_bar = tqdm(data_loaders[phase])
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                count += len(labels)
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                probs = softmax(outputs).detach()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_entropy += np.sum(1 -
                                          normalized_entropy(probs.detach().cpu(), axis=1))
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])

                epoch_loss = running_loss / count
                epoch_acc = running_corrects.double() / count
                epoch_entropy = running_entropy / count
                epoch_avg_max = running_maxes / count
                progress_str = f'{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f} Avg. conf: {epoch_entropy:.2f} Avg. max. prob: {epoch_avg_max:.2f}'
                progress_bar.set_description(progress_str)
            # print(progress_str, flush=True)
            if writer:
                writer.add_scalar(f"Acc/{phase}", epoch_acc, epoch)
                writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            if phase == "val" and epoch_loss < min_val_loss and save_model_filename is not None:
                min_val_loss = epoch_loss
                torch.save(model, save_model_filename)
                print(f"Checkpoint with val_loss = {epoch_loss:.2f} saved.")


def train_unet_model(model, num_epochs, optimizer, criterion, data_loaders, device, writer, save_model_filename=None):
    """Trains a U-Net segmentation model"""
    softmax = nn.Softmax(dim=1)
    model.to(device)
    max_iou = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 10, flush=True)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_corrects = 0
            running_maxes = 0.0
            losses = []
            ious = []
            numel = 0
            progress_bar = tqdm(data_loaders[phase])
            optimizer.zero_grad()
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                numel += labels.numel()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                probs = softmax(outputs).detach()
                # statistics
                running_corrects += torch.sum(preds == labels.data)
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])

                losses.append(loss.item())
                iou = utils.metrics.iou(preds, labels, 4).item()
                ious.append(iou)
                epoch_iou = np.mean(ious)
                epoch_loss = np.mean(losses)
                epoch_acc = running_corrects.double() / numel
                # epoch_entropy = running_entropy / count
                epoch_avg_max = running_maxes / numel
                progress_str = f'{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f} IOU: {epoch_iou:.2f} Avg. max. prob: {epoch_avg_max:.2f}'
                progress_bar.set_description(progress_str)
            writer.add_scalar(f"Acc/{phase}", epoch_acc, epoch)
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"IOU/{phase}", epoch_iou, epoch)

            if phase == "val" and epoch_iou > max_iou and save_model_filename is not None:
                max_iou = epoch_iou
                torch.save(model, save_model_filename)
                print(
                    f"Checkpoint with val IOU = {epoch_iou:.3f} saved.")


def run_validation(model, data_loader, test_progress: Progress, device, mc_dropout_iters=0):
    """Validates classification model"""
    softmax = nn.Softmax(dim=1)
    progress_bar = tqdm(data_loader)
    count = 0
    running_corrects = 0
    model = model.to(device)
    softmax = torch.nn.Softmax(dim=1)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        count += len(labels)
        model.eval()
        with torch.no_grad():
            logits = model(inputs).cpu()
        probs = softmax(logits)
        _, preds = torch.max(logits, 1)
        running_corrects += np.count_nonzero(preds == labels)
        if mc_dropout_iters > 0:
            mc_means, mc_vars = utils.mc_dropout.mc_dropout(
                model, inputs, logits.shape[-1:], T=mc_dropout_iters)
            batch_nll = torch.tensor([0])
            test_progress.update_mcd(mc_means, mc_vars, batch_nll)

        test_progress.update(preds, labels, probs, logits)
        progress_bar.set_description(
            f"Avg. acc.: {100*running_corrects/count:.2f}")

    test_progress.logits = np.concatenate(test_progress.logits)
    test_progress.probs = np.concatenate(test_progress.probs)
    test_progress.predictions = np.concatenate(test_progress.predictions)
    if mc_dropout_iters > 0:
        test_progress.dropout_outputs = np.concatenate(
            test_progress.dropout_outputs)
    return test_progress


def compute_segmentation_loss_weights(dataset: dict, num_classes):
    """Computes weights for classes in the given dataset. Weights are defined as sqrt(total count of pixels / count of pixels for the class)"""
    counts = np.zeros(num_classes, dtype="int64")
    for _, gt in dataset:
        for i in range(num_classes):
            counts[i] += (gt == i).sum()
    all_count = counts.sum()
    weights = [np.sqrt(all_count / count) if count >
               0 else 1.0 for count in counts]
    return weights
