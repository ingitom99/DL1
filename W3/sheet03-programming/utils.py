import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from matplotlib import pyplot as plt

# Specifiy the LeNet5 architecture
class Lenet5(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, latent_dim=32):
        super().__init__()
        self.feature_extractor =  nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(3,  8, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(8,  16, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16, latent_dim),
        )
        self.lin2 = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.lin2(x)
        return x


def show_samples(dataset: VisionDataset):
    h, w = 5, 10
    fig, ax = plt.subplots(h, w)
    fig.set_size_inches((w, h))
    ax = ax.ravel()
    for i in range(h * w):
        img, label = dataset[i]
        ax[i].imshow(torch.permute(img, (1, 2, 0)), cmap='gray')
        ax[i].axis('off')
    plt.show()


def train_one_epoch(
        model: torch.nn.Sequential,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> [float, float]:

    # put the model into the training mode
    model.train()
    
    losses = []
    predictions = []
    labels = []

    for x, y in train_loader:
        
        # forward pass
        logits = model(x)
        
        loss = criterion(logits, y)

        # do gradient updates
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        
        # collect statistics
        prediction = torch.argmax(logits.detach(), dim=-1)
        predictions.append(prediction)
        labels.append(y)
        
        losses.append(loss.detach())
                
        
    epoch_loss = torch.mean(torch.tensor(losses))

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    accuracy = torch.sum(predictions == labels) / len(predictions)

    return float(epoch_loss), float(accuracy)

@torch.no_grad()
def accuracy(model: torch.nn.Sequential, data_loader: DataLoader) -> [float, float]:

    count = 0
    num_correct = 0

    for x, y in data_loader:
        
        logits = model(x)
        
        yh = torch.argmax(logits, dim=1)
                
        num_correct += (y==yh).float().sum()
        count += x.shape[0]
        
    return float(num_correct / count)