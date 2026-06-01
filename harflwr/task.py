import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

WINDOW_SIZE_SECONDS = 5
SAMPLING_RATE_HZ = 20
WINDOW_SIZE = WINDOW_SIZE_SECONDS * SAMPLING_RATE_HZ

# Claude used for model classes for advice on structure and debugging
class HARNetCNN(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class HARNetMLP(nn.Module):

    def __init__(self, num_channels, num_classes, window_size):
        super().__init__()

        input_dim = num_channels * window_size

        self.features = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name, num_channels, num_classes, window_size):
    if model_name == "cnn":
        model = HARNetCNN(num_channels, num_classes)
        return model

    if model_name == "mlp":
        model = HARNetMLP(num_channels, num_classes, window_size)
        return model


def train_fn(model, trainloader, epochs, device, learning_rate):
    if len(trainloader) == 0:
        return 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    total_loss = 0.0
    total_examples = 0

    for epoch in range(epochs):
        for xb, yb in trainloader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            predictions = model(xb)
            loss = criterion(predictions, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_examples += yb.size(0)

    average_loss = total_loss / total_examples
    return average_loss


def evaluate_fn(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total_examples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * yb.size(0)
            correct += (preds == yb).sum().item()
            total_examples += yb.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    average_loss = total_loss / total_examples
    accuracy = correct / total_examples
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_examples": total_examples,
        "y_true": np.asarray(all_labels),
        "y_pred": np.asarray(all_preds),
    }