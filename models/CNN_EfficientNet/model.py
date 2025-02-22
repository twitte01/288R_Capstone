import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    # Load EfficientNet-B0 with pretrained ImageNet weights
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Modify the classifier to match our dataset
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)  # Output probabilities
    )

    return model
