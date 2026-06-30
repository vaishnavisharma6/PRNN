# ============================================
# Phase 5.15: Vision Transformer (ViT)
# PlantVillage classification from scratch
# ============================================

import math
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ------------------------------
# 1. Reproducibility and device
# ------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------
# 2. Dataset paths
# ------------------------------
# Change this to your dataset folder.
# The folder should contain class subfolders directly inside it.
#
# Example:
# /kaggle/input/plantvillagedataset/color
# or
# /content/PlantVillage
#
DATA_DIR = "/kaggle/input/plantvillagedataset/color"


# ------------------------------
# 3. Image transforms
# ------------------------------
# Resize all images to 128x128 as required.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ------------------------------
# 4. Load dataset using ImageFolder
# ------------------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

num_classes = len(full_dataset.classes)
class_names = full_dataset.classes

print("Number of classes:", num_classes)
print("Some classes:", class_names[:5])
print("Total images:", len(full_dataset))


# ------------------------------
# 5. Train / val / test split
# ------------------------------
# Here we do 70 / 15 / 15
dataset_size = len(full_dataset)
train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("\nDataset split:")
print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))


# ------------------------------
# 6. DataLoaders
# ------------------------------
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# ------------------------------
# 7. Visualize one batch
# ------------------------------
images, labels = next(iter(train_loader))
print("\nBatch shape:", images.shape)   # (B, 3, 128, 128)
print("Labels shape:", labels.shape)


# ------------------------------
# 8. Patch embedding module
# ------------------------------
class PatchEmbedding(nn.Module):
    """
    Converts image into a sequence of flattened patches,
    then linearly projects each patch to embedding dimension.
    """

    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection from flattened patch -> embedding
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape

        # Split into patches using unfold
        # After unfold:
        # x -> (B, C, n_patches_h, n_patches_w, patch_h, patch_w)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # Rearrange dimensions
        # (B, C, 8, 8, 16, 16) -> (B, 8, 8, C, 16, 16)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()

        # Flatten each patch
        # -> (B, 64, 3*16*16)
        x = x.view(B, self.num_patches, self.patch_dim)

        # Project to embedding dimension
        x = self.proj(x)   # (B, 64, embed_dim)

        return x


# ------------------------------
# 9. Transformer encoder block
# ------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feedforward with residual connection
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x


# ------------------------------
# 10. Vision Transformer
# ------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=38,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for class token + all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 3, 128, 128)
        """
        B = x.shape[0]

        # Patch embeddings -> (B, 64, embed_dim)
        x = self.patch_embed(x)

        # Expand class token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)

        # Concatenate class token at front
        x = torch.cat((cls_tokens, x), dim=1)           # (B, 65, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        # Use class token output
        cls_output = x[:, 0]                            # (B, embed_dim)
        logits = self.head(cls_output)                  # (B, num_classes)

        return logits


# ------------------------------
# 11. Simple CNN for parameter comparison
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------
# 12. Helper: parameter count
# ------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------
# 13. Build models
# ------------------------------
vit_model = VisionTransformer(
    img_size=128,
    patch_size=16,
    in_channels=3,
    num_classes=num_classes,
    embed_dim=128,
    depth=4,
    num_heads=4,
    mlp_dim=256,
    dropout=0.1
).to(device)

cnn_model = SimpleCNN(num_classes=num_classes).to(device)

print("\nParameter counts:")
print("ViT parameters :", count_parameters(vit_model))
print("CNN parameters :", count_parameters(cnn_model))


# ------------------------------
# 14. Training utilities
# ------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss, acc


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
    return test_loss, test_acc


# ------------------------------
# 15. Train the ViT
# ------------------------------
print("\nTraining Vision Transformer...\n")
vit_model, vit_train_losses, vit_val_losses, vit_train_accs, vit_val_accs = train_model(
    vit_model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3
)

vit_test_loss, vit_test_acc = evaluate_test(vit_model, test_loader)

print("\nViT Test Loss:", vit_test_loss)
print("ViT Test Accuracy:", vit_test_acc)


# ------------------------------
# 16. Plot training curves
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(vit_train_losses, label="Train Loss")
plt.plot(vit_val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ViT Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(vit_train_accs, label="Train Accuracy")
plt.plot(vit_val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ViT Accuracy Curves")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------
# 17. Show patchifying on one image
# ------------------------------
sample_img, sample_label = full_dataset[0]
sample_img_batch = sample_img.unsqueeze(0).to(device)

patch_embed = PatchEmbedding(img_size=128, patch_size=16, in_channels=3, embed_dim=128).to(device)

with torch.no_grad():
    patch_tokens = patch_embed(sample_img_batch)

print("\nOne image shape:", sample_img.shape)           # (3, 128, 128)
print("Patch token shape:", patch_tokens.shape)        # (1, 64, 128)
print("Number of patches:", patch_embed.num_patches)
print("Patch flattened dimension:", patch_embed.patch_dim)


# ------------------------------
# 18. Print final parameter comparison
# ------------------------------
vit_params = count_parameters(vit_model)
cnn_params = count_parameters(cnn_model)

print("\n========== PARAMETER EFFICIENCY ==========")
print("ViT Parameters:", vit_params)
print("CNN Parameters:", cnn_params)

if vit_params < cnn_params:
    print("ViT is more parameter-efficient than the CNN.")
else:
    print("CNN is more parameter-efficient than the ViT.")