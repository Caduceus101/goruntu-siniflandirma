import torch.nn as nn
from torchvision import models


def create_model(num_classes):
    model = models.alexnet(pretrained=True)

    # Son katmanın (classifier) sınıf sayısına göre ayarlanması
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model