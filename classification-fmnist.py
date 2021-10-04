# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets

import utils.metrics
import utils.model

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
data_train = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=True, transform=torchvision.transforms.ToTensor())
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=32,
                                                shuffle=True,
                                                )

data_test = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=False, transform=torchvision.transforms.ToTensor())
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=32,
                                               shuffle=False)


# %%
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %%


class NeuralNetwork(nn.Module):
    def __init__(self, p_dropout):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %%
model = NeuralNetwork(p_dropout=0).to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train_progress = utils.model.train_model(
    model, 5, optimizer, criterion, data_loaders, dataset_sizes)


# %%
def plot_uncertainties(progress):
    correct = progress.predictions == progress.labels
    mc_dropout_confidence = 1 - val_progress.dropout_variances / \
        val_progress.dropout_variances.max()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.violinplot([progress.confidences, progress.max_probs, mc_dropout_confidence,
                   progress.confidences[correct], progress.max_probs[correct], mc_dropout_confidence[correct],
                   progress.confidences[~correct], progress.max_probs[~correct], mc_dropout_confidence[~correct]])

    print(
        f"All predictions mean confidence: {np.mean(progress.confidences)}, "
        f"Prob: {np.mean(progress.max_probs)}, "
        f"Var: {mc_dropout_confidence.mean()}")
    print(
        f"Correct predictions mean confidence: {np.mean(progress.confidences[correct])}, "
        f"Prob: {np.mean(progress.max_probs[correct])}, "
        f"Var: {mc_dropout_confidence[correct].mean()}")
    print(
        f"Incorrect predictions mean confidence: {np.mean(progress.confidences[~correct])}, "
        f"Prob: {np.mean(progress.max_probs[~correct])}, "
        f"Var: {mc_dropout_confidence[~correct].mean()}")


# %%
utils.mc_dropout.set_dropout_p(model, model, .5)
val_progress = utils.model.run_validation(
    model, data_loaders["val"], utils.metrics.Progress(), device, use_mc_dropout=True)

# %%
plot_uncertainties(val_progress)

# %%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)
for label, idx in [("MC dropout", np.argsort(val_progress.dropout_variances)[::-1]),
                   ("Confidence", np.argsort(val_progress.confidences)),
                   ("Max prob", np.argsort(val_progress.max_probs))]:
    labels = val_progress.labels[idx]
    predictions = val_progress.dropout_predictions[
        idx] if label == "MC dropout" else val_progress.predictions[idx]
    accs = utils.metrics.roc_stat(labels, predictions, step=10)
    ax.plot(np.linspace(0, 100, len(accs)), accs, label=label)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()


# %%
def get_mc_dropout_predictions_variance(mc_dropout_output):
    mc_predictions = mc_dropout_output.mean(axis=0).argmax(axis=-1)
    mc_var = mc_dropout_output.var(axis=0).sum(axis=-1)
    return mc_predictions, mc_var


# %%
inputs, classes = next(iter(data_loader_test))
x = inputs[4][0]
y = classes[0]
fig, axs = plt.subplots(1, 5, figsize=(15, 5))
softmax = torch.nn.Softmax(dim=1)
model.eval()
for i in range(5):
    axs[i].axis("off")
    with torch.no_grad():
        outputs = model(torch.unsqueeze(x, 0))
    probs = softmax(outputs)
    _, preds = torch.max(outputs, 1)
    confidence = 1-utils.metrics.normalized_entropy(probs, axis=1)
    max_prob = torch.max(probs, dim=1)
    mc_output = utils.mc_dropout.mc_dropout(model, torch.unsqueeze(x, 0))
    mc_prediction, mc_var = get_mc_dropout_predictions_variance(mc_output)
    axs[i].set_title(f'''Confidence: {confidence[0]:.4f}
    Max prob: {max_prob[0][0]:.4f}
    MC variance: {mc_var[0]:.4f}
    Predicted: {class_names[max_prob[1][0]]}
    ''')
    axs[i].imshow(x, cmap="gray")
    # x[0] = torch.tensor(rotate(x[0], -10))
    x += .5 * torch.rand_like(x)

# %%
num_rows = 3
num_cols = 4
inputs, classes = next(iter(data_loader_test))
inputs = inputs[:num_rows*num_cols]
classes = classes[:num_rows*num_cols]


with torch.no_grad():
    predictions = softmax(model(inputs))
confidences = 1 - utils.metrics.normalized_entropy(predictions, axis=1)
mc_output = utils.mc_dropout.mc_dropout(model, inputs)
mc_prediction, mc_var = get_mc_dropout_predictions_variance(mc_output)

r_index = 0
c_index = 0
fig, axs = plt.subplots(num_rows,  num_cols, figsize=(15, 12))
for i, (x, y) in enumerate(zip(inputs, classes)):
    ax = axs[r_index][c_index]
    ax.axis("off")
    ax.imshow(x[0], cmap="gray")
    predicted = torch.argmax(predictions[i])
    ax.set_title(
        f"""Actual: {class_names[y]}
        Predicted: {class_names[predicted]}
        Confidence: {confidences[i]:.4f}
        Pred max: {torch.max(predictions[i]):.4f}
        Mc dropout var.: {mc_var[i]:.4f}"""
    )
    c_index += 1
    if c_index >= num_cols:
        r_index += 1
        c_index = 0
plt.subplots_adjust(hspace=.5)

# %%
