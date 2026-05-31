import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataset import load_dataset, make_windows, CHORD_TO_IDX
from model import ChordModel


def main():
    torch.cuda.empty_cache()
    print("Loading dataset...")
    try:
        X = np.load("X.npy", mmap_mode="r")
        y = np.load("y.npy", mmap_mode="r")
    except FileNotFoundError:
            print("Preprocessed dataset not found. Creating dataset from raw files...")
            X, y = load_dataset(r'McGill-Billboard')
            y = np.array([CHORD_TO_IDX[chord] for chord in y])
            X_w, y_w = make_windows(X, y)
            np.save('X.npy', X_w)
            np.save('y.npy', y_w)
            print("Dataset saved to disk.")

            X = X_w
            y = y_w 
            print("Dataset loaded and preprocessed.")
            
    X = torch.from_numpy(np.array(X)).float()
    y = torch.from_numpy(np.array(y)).long()

    dataset = TensorDataset(X, y)

    print("Samples:", len(dataset))


    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(
        train_ds,
        prefetch_factor=4,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        prefetch_factor=4,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    model = ChordModel(num_classes=24).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')


    os.makedirs("weights", exist_ok=True)


    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"[E{epoch+1}] batch {batch_idx} loss={loss.item():.4f}")


        avg_loss = total_loss / len(train_loader)


        # validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch).argmax(dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Val acc: {acc:.4f}\n")


        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
            "acc": acc
        }, f"weights/chord_epoch_{epoch+1}.pth")

        print(f"Saved epoch {epoch+1}\n")


if __name__ == "__main__":
    main()