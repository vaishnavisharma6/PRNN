
import kagglehub
abdallahalidev_plantvillage_dataset_path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')

print('Data source import complete.')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

device = "cuda"
print("Device:", device)
from sklearn.decomposition import PCA

DATA_DIR = "/kaggle/input/plantvillage-dataset/plantvillage dataset/color"

print("Folders inside dataset:")
print(os.listdir(DATA_DIR)[:10])

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

num_classes = len(full_dataset.classes)

print("Number of classes:", num_classes)
print("Total images:", len(full_dataset))
print("First few classes:", full_dataset.classes[:10])

val_ratio = 0.2

val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.classifier(x)

    def extract_features(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.relu(self.fc1(x))

model = CNN(num_classes).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5  # increase later if needed

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()

embeddings = []
true_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        feats = model.extract_features(images)

        embeddings.append(feats.cpu().numpy())
        true_labels.append(labels.numpy())

embeddings = np.concatenate(embeddings, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

print("Embeddings shape:", embeddings.shape)

# Clean and scale embeddings safely before GMM
X = np.asarray(embeddings, dtype=np.float32)
y = np.asarray(true_labels)

# Remove rows containing NaN/Inf, and keep labels aligned
valid_rows = np.isfinite(X).all(axis=1)
X = X[valid_rows]
y = y[valid_rows]

# Remove zero-variance / constant feature dimensions
feature_std = X.std(axis=0)
non_constant_features = feature_std > 1e-8
X = X[:, non_constant_features]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)


n_pca = min(50, X.shape[0] - 1, X.shape[1])
if n_pca >= 2:
    pca = PCA(n_components=n_pca, random_state=42)
    X_gmm = pca.fit_transform(X)
else:
    X_gmm = X

true_labels_clean = y

print("Original embeddings shape:", embeddings.shape)
print("Clean embeddings shape:", X_gmm.shape)
print("Number of classes:", num_classes)
print("Unique true labels in validation set:", len(np.unique(true_labels_clean)))

unique_points = len(np.unique(X_gmm, axis=0))
safe_components = min(num_classes, X_gmm.shape[0], unique_points)

if safe_components < 2:
    raise ValueError(
        f"Only {safe_components} usable component(s). Check DATA_DIR and embeddings; clustering needs at least 2 distinct points."
    )

gmm = GaussianMixture(
    n_components=safe_components,
    covariance_type="diag",
    reg_covar=1e-3,      # main fix: prevents ill-defined covariance
    random_state=42,
    n_init=5,
    max_iter=500
)

cluster_labels = gmm.fit_predict(X_gmm)

print("GMM components used:", safe_components)
print("Cluster counts:", np.bincount(cluster_labels))

ari = adjusted_rand_score(true_labels_clean, cluster_labels)

print("\n===== FINAL RESULT =====")
print("Adjusted Rand Index:", ari)

