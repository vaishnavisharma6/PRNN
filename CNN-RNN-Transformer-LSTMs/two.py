import kagglehub
import os
from pathlib import Path

#I ran this code on colab to prevent downloading huge dataset
abdallahalidev_plantvillage_dataset_path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')

print('Data source import complete.')

print("Downloaded path:")
print(abdallahalidev_plantvillage_dataset_path)

base_path = Path(abdallahalidev_plantvillage_dataset_path)
print("\nExists?", base_path.exists())

print("\nTop-level contents:")
for p in base_path.iterdir():
    print(p)

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Dataset path
# -------------------------
base_path = Path("/kaggle/input/plantvillage-dataset/plantvillage dataset")
print("Base path:", base_path)

if not base_path.exists():
    raise ValueError(f"Base path does not exist: {base_path}")

# Choose one: "color", "grayscale", "segmented"
dataset_root = base_path / "color"
print("Using dataset root:", dataset_root)

if not dataset_root.exists():
    raise ValueError(f"Dataset root does not exist: {dataset_root}")

# -------------------------
# Helper functions
# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image_file(path):
    return Path(path).suffix.lower() in IMG_EXTS

def collect_samples_from_class_root(root):
    """
    Expected structure:
    root/
        class_1/
            xxx.jpg
            yyy.jpg
        class_2/
            zzz.jpg
    Returns:
        [(img_path, class_name), ...]
    """
    root = Path(root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if len(class_dirs) == 0:
        raise ValueError(f"No class folders found inside {root}")

    samples = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and is_image_file(img_path):
                samples.append((str(img_path), class_name))
    return samples

# -------------------------
# Collect all samples
# -------------------------
samples = collect_samples_from_class_root(dataset_root)

print("Total samples found:", len(samples))
print("First 5 samples:")
for s in samples[:5]:
    print(s)

if len(samples) == 0:
    raise ValueError(f"No images found inside {dataset_root}")

# -------------------------
# Encode labels
# -------------------------
classes = sorted(list({label for _, label in samples}))
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

print("Number of classes:", len(classes))
print("First 10 classes:", classes[:10])

paths = np.array([p for p, y in samples])
labels_str = np.array([y for p, y in samples])
labels = np.array([class_to_idx[y] for y in labels_str])

# -------------------------
# Stratified split: 70 / 15 / 15
# -------------------------
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)

print("\nSplit sizes:")
print("Train:", len(train_paths))
print("Val  :", len(val_paths))
print("Test :", len(test_paths))

# -------------------------
# Transforms
# -------------------------
IMG_SIZE = 128

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -------------------------
# Dataset class
# -------------------------
class PlantVillageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = int(self.labels[idx])

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# -------------------------
# Create datasets
# -------------------------
train_dataset = PlantVillageDataset(train_paths, train_labels, transform=train_transform)
val_dataset   = PlantVillageDataset(val_paths, val_labels, transform=eval_transform)
test_dataset  = PlantVillageDataset(test_paths, test_labels, transform=eval_transform)

print("\nDataset sizes:")
print("Train dataset:", len(train_dataset))
print("Val dataset  :", len(val_dataset))
print("Test dataset :", len(test_dataset))

# -------------------------
# DataLoaders
# -------------------------
BATCH_SIZE = 32
num_workers = 2 if torch.cuda.is_available() else 0
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

# -------------------------
# Sanity check
# -------------------------
images, targets = next(iter(train_loader))
print("\nOne batch shape:")
print("Images:", images.shape)
print("Labels:", targets.shape)

print("\nClass mapping example:")
for i in range(min(10, len(classes))):
    print(i, "->", idx_to_class[i])

class PlantVillageClassificationDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        y = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, y

train_ds_cls = PlantVillageClassificationDataset(train_paths, train_labels, transform=train_transform)
val_ds_cls   = PlantVillageClassificationDataset(val_paths, val_labels, transform=eval_transform)
test_ds_cls  = PlantVillageClassificationDataset(test_paths, test_labels, transform=eval_transform)

BATCH_SIZE = 64

train_loader_cls = DataLoader(train_ds_cls, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader_cls   = DataLoader(val_ds_cls, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader_cls  = DataLoader(test_ds_cls, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Classification dataloaders ready.")

class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # 128 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 128 -> 64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 64 -> 32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 32 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 32 -> 16

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# 16 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


num_classes = len(classes)
model_cls = PlantCNN(num_classes=num_classes).to(device)
print(model_cls)

# ============================================================
# 3. PRINT TENSOR SHAPES THROUGH EACH LAYER
# ============================================================

def print_tensor_shapes(model, input_shape=(1, 3, 128, 128)):
    x = torch.randn(input_shape).to(device)
    print("Input shape:", tuple(x.shape))

    current = x
    for i, layer in enumerate(model.features):
        current = layer(current)
        print(f"features[{i:02d}] ({layer.__class__.__name__:<12}) -> {tuple(current.shape)}")

    for i, layer in enumerate(model.classifier):
        current = layer(current)
        print(f"classifier[{i:02d}] ({layer.__class__.__name__:<12}) -> {tuple(current.shape)}")

print_tensor_shapes(model_cls, input_shape=(1, 3, 128, 128))

def compute_receptive_field(model):
    """
    Computes theoretical receptive field (RF) and effective jump/stride.
    Formula:
      rf_out = rf_in + (k - 1) * jump_in
      jump_out = jump_in * stride
    """
    rf = 1
    jump = 1
    out_size = IMG_SIZE

    print(f"{'Layer':<20} {'k':<5} {'s':<5} {'p':<5} {'RF':<8} {'Jump':<8}")

    for layer in model.features:
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            if isinstance(layer.kernel_size, tuple):
                k = layer.kernel_size[0]
            else:
                k = layer.kernel_size

            if isinstance(layer.stride, tuple):
                s = layer.stride[0]
            else:
                s = layer.stride if layer.stride is not None else k

            if isinstance(layer.padding, tuple):
                p = layer.padding[0]
            else:
                p = layer.padding

            rf = rf + (k - 1) * jump
            jump = jump * s

            out_size = (out_size + 2*p - k) // s + 1

            print(f"{layer.__class__.__name__:<20} {k:<5} {s:<5} {p:<5} {rf:<8} {jump:<8}")

    print("\nFinal feature map size:", out_size, "x", out_size)
    print("Theoretical receptive field of one activation in final feature map:", rf)
    return rf
rf = compute_receptive_field(model_cls)


def train_one_epoch_classifier(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    preds_all, targets_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.detach().cpu().numpy())
        targets_all.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(targets_all, preds_all)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_classifier(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, targets_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.cpu().numpy())
        targets_all.extend(targets.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(targets_all, preds_all)
    return epoch_loss, epoch_acc    

criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=1e-3)

EPOCHS_CLS = 8

best_val_acc = 0.0
best_cls_path = "/kaggle/working/best_plant_cnn_classifier.pth"

for epoch in range(EPOCHS_CLS):
    train_loss, train_acc = train_one_epoch_classifier(model_cls, train_loader_cls, optimizer_cls, criterion_cls)
    val_loss, val_acc = evaluate_classifier(model_cls, val_loader_cls, criterion_cls)

    print(f"Epoch [{epoch+1}/{EPOCHS_CLS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_cls.state_dict(), best_cls_path)

print("\nBest validation accuracy:", best_val_acc)

model_cls.load_state_dict(torch.load(best_cls_path, map_location=device))
test_loss, test_acc = evaluate_classifier(model_cls, test_loader_cls, criterion_cls)
print("Test accuracy:", test_acc)        


# ============================================================
# 6. SYNTHETIC SEVERITY TARGET
# ============================================================

@lru_cache(maxsize=50000)
def compute_severity_percentage(image_path):
    """
    Synthetic severity target:
    Estimate how much of the leaf area looks diseased.

    Heuristic:
    1. Find leaf region using green OR non-black object pixels.
    2. Inside leaf, count brown/yellow/dark lesion-like pixels.
    3. severity % = lesion_pixels / leaf_pixels * 100

    This is a heuristic target, not ground-truth annotation.
    """
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.uint8)

    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    # Background is often black in PlantVillage
    non_black = (r > 15) | (g > 15) | (b > 15)

    # Approximate leaf mask:
    # either green-ish OR any non-black object
    greenish = (g > r * 0.9) & (g > b * 0.9) & (g > 25)
    leaf_mask = non_black | greenish

    # Lesion-like pixels:
    # brown/yellow/dark regions commonly associated with disease spots
    brownish = (r > g) & (g > b) & (r > 40)
    yellowish = (r > 100) & (g > 100) & (b < 120)
    dark_lesion = (r < 90) & (g < 90) & (b < 90) & non_black

    lesion_mask = (brownish | yellowish | dark_lesion) & leaf_mask

    leaf_area = leaf_mask.sum()
    lesion_area = lesion_mask.sum()

    if leaf_area == 0:
        return 0.0

    severity = 100.0 * lesion_area / leaf_area
    severity = float(np.clip(severity, 0.0, 100.0))
    return severity


# ============================================================
# 7. REGRESSION DATASET
# ============================================================

class PlantVillageRegressionDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        severity = compute_severity_percentage(path)
        target = torch.tensor([severity], dtype=torch.float32)

        return img, target

train_ds_reg = PlantVillageRegressionDataset(train_paths, transform=train_transform)
val_ds_reg   = PlantVillageRegressionDataset(val_paths, transform=eval_transform)
test_ds_reg  = PlantVillageRegressionDataset(test_paths, transform=eval_transform)

train_loader_reg = DataLoader(train_ds_reg, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader_reg   = DataLoader(val_ds_reg, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader_reg  = DataLoader(test_ds_reg, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Regression dataloaders ready.")

# 8. CNN REGRESSION MODEL
# Same CNN backbone, last layer changed to 1 output
# ============================================================

class PlantCNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)   # single continuous output
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

model_reg = PlantCNNRegressor().to(device)
print(model_reg)

def train_one_epoch_regression(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    preds_all, targets_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.detach().cpu().numpy().reshape(-1))
        targets_all.extend(targets.detach().cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae = mean_absolute_error(targets_all, preds_all)
    return epoch_loss, epoch_mae

@torch.no_grad()
def evaluate_regression(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, targets_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.cpu().numpy().reshape(-1))
        targets_all.extend(targets.cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae = mean_absolute_error(targets_all, preds_all)
    return epoch_loss, epoch_mae


criterion_reg = nn.MSELoss()
optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=1e-3)

EPOCHS_REG = 8
best_val_mae = float("inf")
best_reg_path = "/kaggle/working/best_plant_cnn_regressor.pth"

for epoch in range(EPOCHS_REG):
    train_loss, train_mae = train_one_epoch_regression(model_reg, train_loader_reg, optimizer_reg, criterion_reg)
    val_loss, val_mae = evaluate_regression(model_reg, val_loader_reg, criterion_reg)
    print(f"Epoch [{epoch+1}/{EPOCHS_REG}] "
          f"Train MSE: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
          f"Val MSE: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model_reg.state_dict(), best_reg_path)

print("\nBest validation MAE:", best_val_mae)

model_reg.load_state_dict(torch.load(best_reg_path, map_location=device))
test_mse, test_mae = evaluate_regression(model_reg, test_loader_reg, criterion_reg)

print("Test MSE:", test_mse)
print("Test MAE:", test_mae)


# ============================================================
# 10. SHIFT TRANSFORM FOR TRANSLATION INVARIANCE
# ============================================================

class ShiftRightDown:
    def __init__(self, shift_x=5, shift_y=5):
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, img):
        # img is PIL image
        img = img.resize((IMG_SIZE, IMG_SIZE))
        # affine with only translation
        img = TF.affine(
            img,
            angle=0.0,
            translate=(self.shift_x, self.shift_y),
            scale=1.0,
            shear=[0.0, 0.0],
            fill=0
        )
        img = TF.to_tensor(img)
        return img


baseline_transform_100 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

shifted_transform_100 = ShiftRightDown(shift_x=5, shift_y=5)

# pick exactly 100 test images
rng = np.random.default_rng(SEED)
subset_indices = rng.choice(len(test_paths), size=min(100, len(test_paths)), replace=False)

test_paths_100 = test_paths[subset_indices]
test_labels_100 = test_labels[subset_indices]

baseline_100_ds = PlantVillageClassificationDataset(test_paths_100, test_labels_100, transform=baseline_transform_100)
shifted_100_ds  = PlantVillageClassificationDataset(test_paths_100, test_labels_100, transform=shifted_transform_100)

baseline_100_loader = DataLoader(baseline_100_ds, batch_size=32, shuffle=False, num_workers=2)
shifted_100_loader  = DataLoader(shifted_100_ds, batch_size=32, shuffle=False, num_workers=2)

# ============================================================
# 11. BASELINE ACCURACY VS SHIFTED ACCURACY
# ============================================================

@torch.no_grad()
def get_accuracy(model, loader):
    model.eval()
    preds_all, targets_all = [], []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.cpu().numpy())
        targets_all.extend(targets.cpu().numpy())

    return accuracy_score(targets_all, preds_all)

# load best classifier if not already loaded
model_cls.load_state_dict(torch.load(best_cls_path, map_location=device))

baseline_acc = get_accuracy(model_cls, baseline_100_loader)
shifted_acc = get_accuracy(model_cls, shifted_100_loader)
drop = baseline_acc - shifted_acc

print(f"Baseline accuracy on 100 test images : {baseline_acc:.4f}")
print(f"Shifted accuracy on 100 test images  : {shifted_acc:.4f}")
print(f"Performance drop                     : {drop:.4f}")

# ============================================================
# 12. VISUAL CHECK OF SHIFT
# ============================================================

import matplotlib.pyplot as plt

def show_original_and_shifted(paths, n=5):
    fig, axes = plt.subplots(n, 2, figsize=(8, 3*n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        path = paths[i]
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        shifted = TF.affine(img, angle=0.0, translate=(5, 5), scale=1.0, shear=[0.0, 0.0], fill=0)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(shifted)
        axes[i, 1].set_title("Shifted (+5 right, +5 down)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

show_original_and_shifted(test_paths_100[:5], n=5)