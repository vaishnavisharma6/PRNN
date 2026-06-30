
import kagglehub
abdallahalidev_plantvillage_dataset_path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')

print('Data source import complete.')

import os
import random
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm


DATA_ROOT = abdallahalidev_plantvillage_dataset_path

print("Root:", DATA_ROOT)
print("Level 1:", os.listdir(DATA_ROOT))

lvl1 = os.listdir(DATA_ROOT)[0]
path1 = os.path.join(DATA_ROOT, lvl1)

print("Level 2:", os.listdir(path1))

lvl2 = os.listdir(path1)[0]
path2 = os.path.join(path1, lvl2)

print("Level 3:", os.listdir(path2)[:10])

DATA_DIR = os.path.join(
    abdallahalidev_plantvillage_dataset_path,
    "plantvillage dataset",
    "color"
)

base_dataset = datasets.ImageFolder(DATA_DIR)

print("Number of classes:", len(base_dataset.classes))
print("Total images:", len(base_dataset))
print("Sample classes:", base_dataset.classes[:10])

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_DIR = os.path.join(
    abdallahalidev_plantvillage_dataset_path,
    "plantvillage dataset",
    "color"
)

print(os.listdir(DATA_DIR)[:10])

IMG_SIZE = 96
BATCH_SIZE = 64

SIMCLR_EPOCHS = 5
LINEAR_EPOCHS = 10

USE_SUBSET = True
SUBSET_SIZE = 6000
NUM_WORKERS = 2

def make_stratified_subset_indices(imagefolder_dataset, subset_size, seed=42):
    targets = imagefolder_dataset.targets
    class_to_indices = defaultdict(list)

    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    num_classes = len(class_to_indices)
    per_class = max(1, subset_size // num_classes)

    rng = random.Random(seed)
    selected_indices = []

    for label, indices in class_to_indices.items():
        rng.shuffle(indices)
        selected_indices.extend(indices[:per_class])

    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:subset_size]

    return selected_indices

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class SimCLRDataset(Dataset):
    def __init__(self, root, indices=None):
        self.dataset = datasets.ImageFolder(root=root)

        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.dataset[real_idx]

        x1 = simclr_transform(img)
        x2 = simclr_transform(img)

        return x1, x2

base_dataset = datasets.ImageFolder(DATA_DIR)
num_classes = len(base_dataset.classes)

print("Number of classes:", num_classes)
print("Total images:", len(base_dataset))
print("Classes:", base_dataset.classes[:10])

if USE_SUBSET:
    subset_indices = make_stratified_subset_indices(
        base_dataset,
        SUBSET_SIZE,
        seed=SEED
    )
else:
    subset_indices = None

simclr_dataset = SimCLRDataset(DATA_DIR, indices=subset_indices)

train_size = int(0.8 * len(simclr_dataset))
val_size = len(simclr_dataset) - train_size

train_simclr_ds, val_simclr_ds = random_split(
    simclr_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_simclr_loader = DataLoader(
    train_simclr_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_simclr_loader = DataLoader(
    val_simclr_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("SimCLR train batches:", len(train_simclr_loader))
print("SimCLR val batches:", len(val_simclr_loader))

class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        self.encoder = models.resnet18(weights=None)

        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z

    def encode(self, x):
        return self.encoder(x)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N = z1.size(0)

        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N)
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss

simclr_model = SimCLR().to(device)

simclr_loss_fn = InfoNCELoss(temperature=0.5)

simclr_optimizer = torch.optim.Adam(
    simclr_model.parameters(),
    lr=3e-4,
    weight_decay=1e-6
)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train_simclr_epoch(loader):
    simclr_model.train()
    total_loss = 0.0

    for x1, x2 in tqdm(loader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            z1 = simclr_model(x1)
            z2 = simclr_model(x2)
            loss = simclr_loss_fn(z1, z2)

        simclr_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(simclr_optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate_simclr_epoch(loader):
    simclr_model.eval()
    total_loss = 0.0

    for x1, x2 in tqdm(loader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            z1 = simclr_model(x1)
            z2 = simclr_model(x2)
            loss = simclr_loss_fn(z1, z2)

        total_loss += loss.item()

    return total_loss / len(loader)

best_simclr_val_loss = float("inf")

for epoch in range(SIMCLR_EPOCHS):
    train_loss = train_simclr_epoch(train_simclr_loader)
    val_loss = validate_simclr_epoch(val_simclr_loader)

    print(f"\nSimCLR Epoch [{epoch+1}/{SIMCLR_EPOCHS}]")
    print(f"Train InfoNCE Loss: {train_loss:.4f}")
    print(f"Val InfoNCE Loss:   {val_loss:.4f}")

    if val_loss < best_simclr_val_loss:
        best_simclr_val_loss = val_loss
        torch.save(simclr_model.state_dict(), "simclr_encoder.pth")

print("Best SimCLR Validation Loss:", best_simclr_val_loss)

labeled_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=eval_transform
)

if USE_SUBSET:
    labeled_dataset = Subset(labeled_dataset, subset_indices)

print("Linear eval dataset size:", len(labeled_dataset))

labels = []

for i in range(len(labeled_dataset)):
    _, y = labeled_dataset[i]
    labels.append(y)

print("Class distribution:")
print(Counter(labels))
print("Unique classes:", len(set(labels)))

total_size = len(labeled_dataset)

train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    labeled_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

linear_eval_simclr = SimCLR().to(device)
linear_eval_simclr.load_state_dict(
    torch.load("simclr_encoder.pth", map_location=device)
)

encoder = linear_eval_simclr.encoder

for param in encoder.parameters():
    param.requires_grad = False

classifier = nn.Linear(512, num_classes).to(device)

criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3)

def train_linear_epoch(loader):
    encoder.eval()
    classifier.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            features = encoder(images)

        logits = classifier(features)
        loss = criterion_cls(logits, labels)

        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total

    return avg_loss, acc

@torch.no_grad()
def evaluate_linear_epoch(loader):
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = encoder(images)
        logits = classifier(features)
        loss = criterion_cls(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total

    return avg_loss, acc

best_val_acc = 0.0

for epoch in range(LINEAR_EPOCHS):
    train_loss, train_acc = train_linear_epoch(train_loader)
    val_loss, val_acc = evaluate_linear_epoch(val_loader)

    print(f"\nLinear Eval Epoch [{epoch+1}/{LINEAR_EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(classifier.state_dict(), "linear_head.pth")

print("Best Linear Validation Accuracy:", best_val_acc)

classifier.load_state_dict(
    torch.load("linear_head.pth", map_location=device)
)

test_loss, test_acc = evaluate_linear_epoch(test_loader)

print("\n================ FINAL RESULTS ================")
print(f"Final SimCLR Validation InfoNCE Loss: {best_simclr_val_loss:.4f}")
print(f"Final Linear Evaluation Test Loss:    {test_loss:.4f}")
print(f"Final Linear Evaluation Test Acc:     {test_acc:.4f}")