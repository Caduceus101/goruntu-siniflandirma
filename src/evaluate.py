import torch
from src.config import device
from src.data_loader import get_data_loaders
from src.model import create_model
from sklearn.metrics import accuracy_score, classification_report


def evaluate():
    train_loader, val_loader, classes = get_data_loaders()

    model = create_model(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Test Doğruluğu (Accuracy):", accuracy_score(all_labels, all_preds))
    print("\nSınıflandırma Raporu:\n", classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    evaluate()