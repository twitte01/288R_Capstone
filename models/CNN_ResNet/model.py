import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    # Load ResNet18 with pretrained ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the fully connected layer to match our dataset
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)  # Output probabilities
    )

    return model
