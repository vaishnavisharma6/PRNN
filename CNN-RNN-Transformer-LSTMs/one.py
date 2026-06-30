import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# 1. Reproducibility and device
# ------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------
# 2. Load and preprocess dataset
# ------------------------------
# Change this path to your file
CSV_PATH = "/Users/vaishnavisharma/prnn/delhi_aqi.csv"


df = pd.read_csv(CSV_PATH)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# ------------------------------
# 3. Select time column and features
# ------------------------------
# Try to automatically find a time column
possible_time_cols = ["datetime", "date", "timestamp", "time", "Date", "Datetime"]
time_col = None
for col in possible_time_cols:
    if col in df.columns:
        time_col = col
        break

# Candidate meteorological / pollution features
candidate_features = [
    'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3'
]

available_features = [col for col in candidate_features if col in df.columns]

if len(available_features) == 0:
    raise ValueError("None of the expected feature columns were found in the CSV.")

target_col = "pm2_5"
if target_col not in available_features:
    raise ValueError(f"Target column '{target_col}' not found in selected features.")

print("\nUsing features:")
print(available_features)

# Keep only needed columns
if time_col is not None:
    df = df[[time_col] + available_features].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
else:
    df = df[available_features].copy()

# Convert to numeric
for col in available_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing rows
df = df.dropna().reset_index(drop=True)

print("Shape after cleanup:", df.shape)


# ------------------------------
# 4. Chronological split
# ------------------------------
# 70% train, 15% val, 15% test
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))


# ------------------------------
# 5. Standardize using train stats only
# ------------------------------
feature_means = train_df[available_features].mean()
feature_stds  = train_df[available_features].std().replace(0, 1)

train_scaled = (train_df[available_features] - feature_means) / feature_stds
val_scaled   = (val_df[available_features] - feature_means) / feature_stds
test_scaled  = (test_df[available_features] - feature_means) / feature_stds

target_idx = available_features.index(target_col)
print("\nTarget column index:", target_idx)


# ------------------------------
# 6. Sequence creation
# ------------------------------
SEQ_LEN = 72  # 72-hour sequence

def create_sequences(dataframe_scaled, seq_len, target_idx):
    data = dataframe_scaled.values.astype(np.float32)
    xs, ys = [], []

    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]                 # shape: (72, num_features)
        y = data[i + seq_len, target_idx]      # next-step pm2_5
        xs.append(x)
        ys.append(y)

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN, target_idx)
X_val, y_val     = create_sequences(val_scaled, SEQ_LEN, target_idx)
X_test, y_test   = create_sequences(test_scaled, SEQ_LEN, target_idx)

print("\nSequence dataset shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)


# ------------------------------
# 7. Dataset and DataLoader
# ------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset   = TimeSeriesDataset(X_val, y_val)
test_dataset  = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ------------------------------
# 8. Single-head scaled dot-product attention
# ------------------------------
class ScaledDotProductAttentionHead(nn.Module):
    def __init__(self, input_dim, d_k):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, d_k)
        self.k_proj = nn.Linear(input_dim, d_k)
        self.v_proj = nn.Linear(input_dim, d_k)
        self.scale = math.sqrt(d_k)

    def forward(self, x):
        """
        x: (B, T, input_dim)

        returns:
            out: (B, T, d_k)
            attn_weights: (B, T, T)
        """
        Q = self.q_proj(x)                      # (B, T, d_k)
        K = self.k_proj(x)                      # (B, T, d_k)
        V = self.v_proj(x)                      # (B, T, d_k)

        # scores = QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # (B, T, T)

        # attention weights
        attn_weights = torch.softmax(scores, dim=-1)                  # (B, T, T)

        # weighted sum
        out = torch.matmul(attn_weights, V)                           # (B, T, d_k)

        return out, attn_weights


# ------------------------------
# 9. Full model
# ------------------------------
class AttentionRegressor(nn.Module):
    def __init__(self, input_dim, d_k=32):
        super().__init__()
        self.attn = ScaledDotProductAttentionHead(input_dim, d_k)
        self.fc = nn.Sequential(
            nn.Linear(d_k, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (B, T, input_dim)

        We use the representation of the LAST time step
        after attention to predict next-step pm2_5.
        """
        attn_out, attn_weights = self.attn(x)        # (B, T, d_k), (B, T, T)
        last_token = attn_out[:, -1, :]              # (B, d_k)
        pred = self.fc(last_token).squeeze(-1)       # (B,)
        return pred, attn_weights


input_dim = len(available_features)
model = AttentionRegressor(input_dim=input_dim, d_k=32).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nModel:")
print(model)


# ------------------------------
# 10. Training and evaluation
# ------------------------------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds, _ = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        count += xb.size(0)

    return total_loss / count


EPOCHS = 200

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = run_epoch(model, train_loader, optimizer=optimizer)
    val_loss = run_epoch(model, val_loader, optimizer=None)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")


# ------------------------------
# 11. Test loss
# ------------------------------
test_loss = run_epoch(model, test_loader, optimizer=None)
print("\nTest Loss:", test_loss)


# ------------------------------
# 12. Plot training and validation loss
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Attention Model Training")
plt.legend()
plt.grid(True)
plt.savefig('trans-loss.jpeg')


# ------------------------------
# 13. Extract 72x72 attention matrix for one test sample
# ------------------------------
model.eval()

sample_idx = 0  # you can change this
x_sample, y_sample = test_dataset[sample_idx]

with torch.no_grad():
    x_input = x_sample.unsqueeze(0).to(device)      # (1, 72, input_dim)
    pred_sample, attn_weights_sample = model(x_input)

# attention matrix for this one sample
attn_matrix = attn_weights_sample[0].cpu().numpy()  # (72, 72)

print("\nSingle test sample:")
print("Input shape:", x_sample.shape)
print("Target:", y_sample.item())
print("Prediction:", pred_sample.item())
print("Attention matrix shape:", attn_matrix.shape)


# ------------------------------
# 14. Plot attention heatmap
# ------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(attn_matrix, aspect="auto", cmap="viridis")
plt.colorbar(label="Attention Weight")
plt.xlabel("Key Time Step")
plt.ylabel("Query Time Step")
plt.title("72 x 72 Attention Weight Matrix for One Test Sample")
plt.savefig('heatmap.jpeg')


# ------------------------------
# 15. Optional: show last-step attention only
# ------------------------------
# This tells you where the final query step is attending
last_row = attn_matrix[-1]

plt.figure(figsize=(10, 4))
plt.plot(np.arange(1, SEQ_LEN + 1), last_row)
plt.xlabel("Input Time Step")
plt.ylabel("Attention Weight")
plt.title("Attention Weights from Final Query Time Step")
plt.grid(True)
plt.show()