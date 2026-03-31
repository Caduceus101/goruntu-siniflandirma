import torch
import torch.nn as nn
import torch.optim as optim
from src.config import device
from src.data_loader import get_data_loaders
from src.model import create_model


def train():
    train_loader, val_loader, classes = get_data_loaders()
    model = create_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Model ağırlıklarını klasöre kaydet
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Eğitim başarıyla tamamlandı ve lokal olarak kaydedildi.")


if __name__ == "__main__":
    train()