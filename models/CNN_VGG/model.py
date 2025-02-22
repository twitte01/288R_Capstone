import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    # Load VGG16 model
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Keep original
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    dummy_input = torch.randn(1, 3, 64, 64)  # Simulate input
    dummy_output = model.features(dummy_input)
    num_features = dummy_output.view(1, -1).size(1)  # Get flattened size

    # Modify classifier dynamically
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),  # Use dynamically computed features
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)  # Output probabilities
    )

    return model

