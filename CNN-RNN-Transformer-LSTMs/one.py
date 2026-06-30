# =========================================================
# APTOS 2019 - Phase 6.16
# Transfer Learning & Freezing
# Full Kaggle-compatible code
# =========================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# -----------------------------
# 1. Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------------
# 2. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 3. Kaggle paths
# -----------------------------
CSV_PATH = "/kaggle/input/competitions/aptos2019-blindness-detection/train.csv"
IMAGE_DIR = "/kaggle/input/competitions/aptos2019-blindness-detection/train_images"

# -----------------------------
# 4. Load CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:")
print(df["diagnosis"].value_counts().sort_index())

# -----------------------------
# 5. Stratified train/val split
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\nTrain size:", len(train_df))
print("Val size  :", len(val_df))

print("\nTrain class counts:")
print(train_df["diagnosis"].value_counts().sort_index())

print("\nVal class counts:")
print(val_df["diagnosis"].value_counts().sort_index())

# -----------------------------
# 6. Dataset
# -----------------------------
class APTOSDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row["id_code"]
        label = int(row["diagnosis"])

        image_path = os.path.join(self.image_dir, image_id + ".png")
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# -----------------------------
# 7. Transforms
# ResNet18 expects ImageNet normalization
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 8. Datasets and loaders
# -----------------------------
train_dataset = APTOSDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset = APTOSDataset(val_df, IMAGE_DIR, transform=val_transform)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# -----------------------------
# 9. Load pretrained ResNet18
# -----------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze all existing parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer for 5 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 5)
model = model.to(device)

# -----------------------------
# 10. Count trainable/frozen parameters
# -----------------------------
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
total_params = trainable_params + frozen_params

print("\nParameter counts:")
print("Trainable parameters:", trainable_params)
print("Frozen parameters   :", frozen_params)
print("Total parameters    :", total_params)

# Optional: print exact trainable layer names
print("\nTrainable layers:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

# -----------------------------
# 11. Loss and optimizer
# Only train the new head
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ----------------# 12. Train function
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

# -----------------------------
# 13. Validation function
# -----------------------------
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

# -----------------------------
# 14. Training loop
# -----------------------------
NUM_EPOCHS = 10

train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, device
    )

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())

# Load best model
model.load_state_dict(best_model_state)

print("\nBest Validation Loss:", best_val_loss)

# -----------------------------
# 15. Final validation evaluation
# -----------------------------
final_val_loss, final_val_acc = validate_one_epoch(
    model, val_loader, criterion, device
)

print("\nFinal Best Model Metrics on Validation Set")
print(f"Validation Loss: {final_val_loss:.4f}")
print(f"Validation Acc : {final_val_acc:.4f}")

# -----------------------------
# 16. Plot validation loss curve
# This is required by the question
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve")
plt.grid(True)
plt.show()

# -----------------------------
# 17. Optional: plot train/val loss together
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker="o", label="Train Loss")
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, marker="s", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 18. Optional: plot train/val accuracy
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_accs, marker="o", label="Train Accuracy")
plt.plot(range(1, NUM_EPOCHS + 1), val_accs, marker="s", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 19. Save best model
# -----------------------------
SAVE_PATH = "/kaggle/working/aptos_resnet18_frozen_head_best.pth"
torch.save(model.state_dict(), SAVE_PATH)
print("\nBest model saved to:", SAVE_PATH)