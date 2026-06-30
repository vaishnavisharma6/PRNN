import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. SETTINGS
# =========================================================

TIME_COL = "date"                                          # change if needed
FEATURE_COLS = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']    # change if needed
TARGET_COL = "pm2_5"

SEQ_LEN = 72
BATCH_SIZE = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 1

LR = 1.0                 # intentionally very large
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# =========================================================
# 2. LOAD DATASET
# =========================================================
df = pd.read_csv('/Users/vaishnavisharma/prnn/delhi_aqi.csv')

print("Columns in dataset:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 3. KEEP ONLY REQUIRED COLUMNS
# =========================================================
df = df[[TIME_COL] + FEATURE_COLS].copy()


# =========================================================
# 4. SORT BY TIME
# =========================================================
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)


# =========================================================
# 5. HANDLE MISSING VALUES
# =========================================================
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

print("\nRows after cleaning:", len(df))


# =========================================================
# 6. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT: 70 / 15 / 15
# =========================================================
n_total = len(df)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_df = df.iloc[:n_train].copy()
val_df = df.iloc[n_train:n_train + n_val].copy()
test_df = df.iloc[n_train + n_val:].copy()

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))


# =========================================================
# 7. SCALE DATA
# =========================================================
train_values = train_df[FEATURE_COLS].values
val_values = val_df[FEATURE_COLS].values
test_values = test_df[FEATURE_COLS].values

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_values)
val_scaled = scaler.transform(val_values)
test_scaled = scaler.transform(test_values)

target_col_idx = FEATURE_COLS.index(TARGET_COL)
print("\nTarget column index:", target_col_idx)


# =========================================================
# 8. CREATE SEQUENCES
# =========================================================
def create_sequences(data_array, seq_len, target_col_idx):
    X = []
    y = []

    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i + seq_len])
        y.append(data_array[i + seq_len, target_col_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


X_train, y_train = create_sequences(train_scaled, SEQ_LEN, target_col_idx)
X_val, y_val = create_sequences(val_scaled, SEQ_LEN, target_col_idx)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN, target_col_idx)

print("\nSequence shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


# =========================================================
# 9. DATASET + DATALOADER
# =========================================================
class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = AirQualityDataset(X_train, y_train)
val_dataset = AirQualityDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================================================
# 10. VANILLA RNN FROM SCRATCH
# =========================================================
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wy = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(self.Wx(x_t) + self.Wh(h))

        out = self.Wy(h)
        return out


input_dim = X_train.shape[2]
model = VanillaRNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(DEVICE)

criterion = nn.MSELoss()

# Intentionally huge learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =========================================================
# 11. TRAINING LOOP TO SHOW EXPLODING GRADIENTS
# =========================================================
train_losses = []
val_losses = []
grad_norms = []
sigma_max_history = []

crashed = False
epoch_of_crash = None
Wh_before_crash = None

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    max_grad_norm_epoch = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        # Check loss before backward
        if torch.isnan(loss) or torch.isinf(loss):
            crashed = True
            epoch_of_crash = epoch + 1
            break

        loss.backward()

        # Total gradient norm over all parameters
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2).item()
                total_grad_norm += param_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5

        max_grad_norm_epoch = max(max_grad_norm_epoch, total_grad_norm)

        optimizer.step()

        # Check parameters after update
        bad_param = False
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                bad_param = True
                break

        if bad_param:
            crashed = True
            epoch_of_crash = epoch + 1
            break

        running_train_loss += loss.item() * X_batch.size(0)

    # Save Wh and singular value before crash/at epoch end
    Wh_matrix = model.Wh.weight.detach().cpu()
    sigma_max = torch.linalg.svdvals(Wh_matrix).max().item()
    sigma_max_history.append(sigma_max)
    grad_norms.append(max_grad_norm_epoch)

    if crashed:
        Wh_before_crash = Wh_matrix.clone()
        print(f"\nTraining became unstable at epoch {epoch_of_crash}")
        break

    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            val_loss = criterion(preds, y_batch)
            running_val_loss += val_loss.item() * X_batch.size(0)

    avg_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"Max Grad Norm: {max_grad_norm_epoch:.6f} | "
        f"Sigma_max(Wh): {sigma_max:.6f}"
    )


# =========================================================
# 12. IF NO CRASH, STILL SAVE FINAL Wh
# =========================================================
if Wh_before_crash is None:
    Wh_before_crash = model.Wh.weight.detach().cpu().clone()

final_sigma_max = torch.linalg.svdvals(Wh_before_crash).max().item()

print("\nLargest singular value of Wh:", final_sigma_max)


# =========================================================
# 13. PLOT TRAIN / VAL LOSS
# =========================================================
if len(train_losses) > 0:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss during high-LR training")
    plt.legend()
    plt.grid(True)
    plt.savefig('highlr.jpeg')


# =========================================================
# 14. PLOT MAX GRADIENT NORM PER EPOCH
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(grad_norms) + 1), grad_norms)
plt.xlabel("Epoch")
plt.ylabel("Max gradient norm")
plt.title("Gradient growth across epochs")
plt.grid(True)
plt.savefig('max-gradient.jpeg')


# =========================================================
# 15. PLOT LARGEST SINGULAR VALUE OF Wh
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(sigma_max_history) + 1), sigma_max_history)
plt.xlabel("Epoch")
plt.ylabel("Largest singular value of Wh")
plt.title("Growth of sigma_max(Wh)")
plt.grid(True)
plt.savefig('sigma.jpeg')