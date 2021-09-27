import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.mc_dropout import mc_dropout
from utils.metrics import Progress, normalized_entropy


def train_model(model, num_epochs, optimizer, criterion, data_loaders, dataset_sizes):
    softmax = nn.Softmax(dim=1)
    precision_holder = []
    for epoch in range(num_epochs):
        precision_holder.append({
            "train": Progress(),
            "val": Progress()
        })
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 10, flush=True)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            running_entropy = 0.0
            running_maxes = 0.0
            count = 0
            progress_bar = tqdm(data_loaders[phase])
            for inputs, labels in progress_bar:
                count += len(labels)
                current_holder = precision_holder[epoch][phase]
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                probs = softmax(outputs)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_entropy += np.sum(1 -
                                          normalized_entropy(probs.detach().numpy(), axis=1))
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])
                current_holder = current_holder.update(
                    preds, labels, probs)

                epoch_loss = running_loss / count
                epoch_acc = running_corrects.double() / count
                epoch_entropy = running_entropy / count
                epoch_avg_max = running_maxes / count
                progress_str = f'{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f} Avg. conf: {epoch_entropy:.2f} Avg. max. prob: {epoch_avg_max:.2f}'
                progress_bar.set_description(progress_str)
    return precision_holder


def run_validation(model, data_loader):
    softmax = nn.Softmax(dim=1)
    test_progress = Progress()
    progress_bar = tqdm(data_loader)
    count = 0
    running_corrects = 0
    for inputs, labels in progress_bar:
        count += len(labels)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        probs = softmax(outputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += np.count_nonzero(preds == labels)
        mc_output = mc_dropout(
            model, inputs).detach().numpy()
        mc_predictions = np.mean(mc_output, axis=0).argmax(axis=-1)
        mc_var = mc_output.var(axis=0).sum(axis=-1)
        test_progress.dropout_predictions = np.append(
            test_progress.dropout_predictions, mc_predictions)
        test_progress.dropout_variances = np.append(
            test_progress.dropout_variances, mc_var)
        test_progress.update(preds, labels, probs)
        progress_bar.set_description(
            f"Avg. acc.: {running_corrects/count:.2f}")
    return test_progress


def get_number_of_classes(model):
    last = list(model.children())[-1]
    try:
        last_ftrs = last.out_features
        return last_ftrs
    except AttributeError:
        return get_number_of_classes(last)
