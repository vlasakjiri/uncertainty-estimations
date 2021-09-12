# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from skimage.transform import rotate
import torchvision.models as models
import torchvision.datasets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# %%
data_train = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=True, transform=torchvision.transforms.ToTensor())
data_loader_train = torch.utils.data.DataLoader(data_train,
                                                batch_size=256,
                                                shuffle=True,
                                                )

data_test = torchvision.datasets.FashionMNIST(
    "fmnist", download=True, train=False, transform=torchvision.transforms.ToTensor())
data_loader_test = torch.utils.data.DataLoader(data_test,
                                               batch_size=256,
                                               shuffle=False)


# %%
dataset_sizes = {"train": len(data_train), "val": len(data_test)}
data_loaders = {"train": data_loader_train, "val": data_loader_test}


# %%
len(data_loader_train), len(data_loader_test)


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
def mc_dropout(model, X, num_classes=10, T=100):
    model.train()
    softmax = nn.Softmax(dim=1)
    out = torch.zeros(T, X.shape[0], num_classes)
    for i in range(T):
        with torch.no_grad():
            out[i] = softmax(model(X))
    return out


# %%
def normalized_entropy(probs, num_classes=10, **kwargs):
    return entropy(probs, **kwargs)/np.log(num_classes)


# %%
model = NeuralNetwork(p_dropout=.5).to("cpu")
print(model)
num_epochs = 15
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        running_entropy = 0.0
        running_maxes = 0.0
        for inputs, labels in data_loaders[phase]:
            optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                maxs, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                probs = softmax(outputs)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_entropy += np.sum(1 -
                                          normalized_entropy(probs.detach().numpy(), axis=1))
                running_maxes += torch.sum(torch.max(probs, dim=1)[0])
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        epoch_entropy = running_entropy / dataset_sizes[phase]
        epoch_avg_max = running_maxes / dataset_sizes[phase]
        print('{} Loss: {:.4f} Acc: {:.4f} Avg. conf: {:.4f} Avg. max. prob: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, epoch_entropy, epoch_avg_max))

# %%
batch_size = 10
bin_loader = torch.utils.data.DataLoader(data_test,
                                         batch_size=batch_size,
                                         shuffle=False)
accs = []
entropies = []
maxes = []
for inputs, labels in bin_loader:
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probs = softmax(outputs)
        _, preds = torch.max(outputs, 1)
        accs.append(torch.sum(preds == labels.data) / batch_size)
        entropies.append(
            np.sum(1-normalized_entropy(probs.detach().numpy(), axis=1)) / batch_size)
        maxes.append(torch.sum(torch.max(probs, dim=1)[0]) / batch_size)


# %%

idxs = np.argsort(accs)
accs = np.array(accs)[idxs]
entropies = np.array(entropies)[idxs]
maxes = np.array(maxes)[idxs]
plt.plot(entropies, accs)
plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0))
# plt.plot(entropies)
# plt.plot(maxes)

# %%
inputs, classes = next(iter(data_loader_test))
x = inputs[0]
y = classes[0]
fig, axs = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axs[i].axis("off")
    with torch.no_grad():
        outputs = model(x)
    probs = softmax(outputs)
    _, preds = torch.max(outputs, 1)
    confidence = 1-normalized_entropy(probs, axis=1)
    max_prob = torch.max(probs, dim=1)
    axs[i].set_title(f'''Confidence: {confidence[0]:4f}
    Max prob: {max_prob[0][0]:4f}
    Predicted: {class_names[max_prob[1][0]]}
    ''')
    axs[i].imshow(x[0], cmap="gray")
    # x[0] = torch.tensor(rotate(x[0], -10))
    x+= .2*torch.rand_like(x)

# %%
num_rows = 3
num_cols = 4
inputs, classes = next(iter(data_loader_test))
inputs = inputs[:num_rows*num_cols]
classes = classes[:num_rows*num_cols]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


with torch.no_grad():
    softmax = nn.Softmax(dim=1)
    predictions = softmax(model(inputs))
uncertainties = normalized_entropy(predictions, axis=1)

r_index = 0
c_index = 0
fig, axs = plt.subplots(num_rows,  num_cols, figsize=(15, 12))
for i, (x, y) in enumerate(zip(inputs, classes)):
    ax = axs[r_index][c_index]
    ax.axis("off")
    ax.imshow(x[0], cmap="gray")
    predicted = torch.argmax(predictions[i])
    ax.set_title(
        f"Actual: {class_names[y]}\nPrediction: {class_names[predicted]}\nConfidence: {1-uncertainties[i]}\n Pred max: {torch.max(predictions[i])}")
    c_index += 1
    if c_index >= num_cols:
        r_index += 1
        c_index = 0
plt.subplots_adjust(hspace=.4)

# %%
