import argparse
import os
import torch
import torchvision
import mlflow
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    num_classes = 4  # crushed, torn, water-damaged, intact-flagged
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(args.data, transform=transform)
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # --- Transfer learning setup: load pre-trained weights, freeze the backbone,
    #     replace the final layer with a head sized to our classes ---
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def train_one_epoch():
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(train_ds)

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total else 0.0

    mlflow.start_run()
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("base_model", "resnet50")
    mlflow.log_param("epochs", args.epochs)

    val_accuracy = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch()
        val_accuracy = evaluate()
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_accuracy={val_accuracy:.4f}")

    os.makedirs("./outputs", exist_ok=True)
    torch.save(model.state_dict(), "./outputs/model.pt")
    mlflow.log_metric("final_val_accuracy", val_accuracy)
    mlflow.end_run()

if __name__ == "__main__":
    main()
