from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import TRAIN_PATH, VAL_PATH, LENGTH, BATCH_SIZE

def get_data_loaders():
    # 1024 boyutuna göre yeniden boyutlandırma ve tensör işlemleri
    train_transforms = transforms.Compose([
        transforms.Resize((LENGTH, LENGTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((LENGTH, LENGTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset.classes