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
from tqdm import tqdm


# %%
device = "cpu"

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


class Progress:
    def __init__(self) -> None:
        self.predictions = np.array([])
        self.labels = np.array([])
        self.max_probs = np.array([])
        self.confidences = np.array([])
        self.dropout_variances = np.array([])

    def update(self, preds, labels, probs):
        self.predictions = np.append(
            self.predictions, preds.numpy())
        self.labels = np.append(
            self.labels, labels.data.numpy())
        self.max_probs = np.append(self.max_probs, torch.max(
            probs, dim=1)[0].detach().numpy())
        self.confidences = np.append(self.confidences,
                                     1 - normalized_entropy(probs.detach().numpy(), axis=1))

    def __str__(self) -> str:
        return f"Predictions: {self.predictions}\nLabels: {self.labels}\nMax probs: {self.max_probs}\n Confidences: {self.confidences}"


# %%


def train_model(model, num_epochs, optimizer, criterion):
    softmax = nn.Softmax(dim=1)
    precision_holder = []
    for epoch in range(num_epochs):
        precision_holder.append({
            "train": Progress(),
            "val": Progress()
        })
        print(f'Epoch {epoch+1}/{num_epochs}')
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
            progress_bar = tqdm(data_loaders[phase])
            for inputs, labels in progress_bar:
                current_holder = precision_holder[epoch][phase]
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    probs = softmax(outputs)
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_entropy += np.sum(1 -
                                              normalized_entropy(probs.detach().numpy(), axis=1))
                    running_maxes += torch.sum(torch.max(probs, dim=1)[0])
                    current_holder = current_holder.update(
                        preds, labels, probs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_entropy = running_entropy / dataset_sizes[phase]
            epoch_avg_max = running_maxes / dataset_sizes[phase]
            progress_bar.set_description('{} Loss: {:.4f} Acc: {:.4f} Avg. conf: {:.4f} Avg. max. prob: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_entropy, epoch_avg_max))
            # print()
    return precision_holder


# %%
model = NeuralNetwork(p_dropout=0.5).to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train_progress = train_model(model, 5, optimizer, criterion)


# %%
def plot_uncertainties(progress):
    correct = progress.predictions == progress.labels
    plt.figure(figsize=(15, 5))
    plt.violinplot([progress.confidences, progress.max_probs,
                    progress.confidences[correct], progress.max_probs[correct],
                    progress.confidences[~correct], progress.max_probs[~correct]])

    print(
        f"All predictions mean confidence: {np.mean(progress.confidences)}, "
        f"Prob: {np.mean(progress.max_probs)}, "
        f"Var: {progress.dropout_variances.mean()}")
    print(
        f"Correct predictions mean confidence: {np.mean(progress.confidences[correct])}, "
        f"Prob: {np.mean(progress.max_probs[correct])}, "
        f"Var: {progress.dropout_variances[correct].mean()}")
    print(
        f"Incorrect predictions mean confidence: {np.mean(progress.confidences[~correct])}, "
        f"Prob: {np.mean(progress.max_probs[~correct])}, "
        f"Var: {progress.dropout_variances[~correct].mean()}")

# %%


def run_validation(model, data_loader):
    softmax = nn.Softmax(dim=1)
    test_progress = Progress()
    for inputs, labels in tqdm(data_loader):
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)
            mc_output = mc_dropout(model, inputs).detach().numpy()
            mc_var = mc_output.var(axis=0).sum(axis=-1)
            test_progress.dropout_variances = np.append(
                test_progress.dropout_variances, mc_var)
            test_progress.update(preds, labels, probs)
    return test_progress


# %%
val_progress = run_validation(model, data_loaders["val"])

# %%
plot_uncertainties(val_progress)

# %%
import matplotlib.ticker as mtick
idx = np.argsort(val_progress.dropout_variances)[::-1]
labels = val_progress.labels[idx]
predictions = val_progress.predictions[idx]
accs=[]
step = 10
for _ in range(len(labels)//step):
    accs.append((labels == predictions).sum()/len(labels))
    labels = labels[step:]
    predictions = predictions[step:]
fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0,100,len(accs)), accs)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
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
    x += .2*torch.rand_like(x)

# %%
num_rows = 3
num_cols = 4
inputs, classes = next(iter(data_loader_test))
inputs = inputs[:num_rows*num_cols]
classes = classes[:num_rows*num_cols]


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
