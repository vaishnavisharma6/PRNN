import math
import copy
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
# 2. Load dataset
# ------------------------------
# Change path as needed
CSV_PATH = "/Users/vaishnavisharma/prnn/delhi_aqi.csv"
# Example:
# CSV_PATH = "/content/delhi_aqi.csv"

df = pd.read_csv(CSV_PATH)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# ------------------------------
# 3. Select columns
# ------------------------------
possible_time_cols = ["datetime", "date", "timestamp", "time", "Date", "Datetime"]
time_col = None
for col in possible_time_cols:
    if col in df.columns:
        time_col = col
        break

candidate_features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
available_features = [col for col in candidate_features if col in df.columns]

if len(available_features) == 0:
    raise ValueError("Expected feature columns not found in dataset.")

target_col = "pm2_5"
if target_col not in available_features:
    raise ValueError("pm2_5 not found in selected features.")

print("\nUsing features:")
print(available_features)

if time_col is not None:
    df = df[[time_col] + available_features].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
else:
    df = df[available_features].copy()

for col in available_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

print("Rows after cleanup:", len(df))


# ------------------------------
# 4. Chronological split
# ------------------------------
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
# 5. Standardization using train only
# ------------------------------
feature_means = train_df[available_features].mean()
feature_stds  = train_df[available_features].std().replace(0, 1)

train_scaled = (train_df[available_features] - feature_means) / feature_stds
val_scaled   = (val_df[available_features] - feature_means) / feature_stds
test_scaled  = (test_df[available_features] - feature_means) / feature_stds

target_idx = available_features.index(target_col)
print("\nTarget column index:", target_idx)


# ------------------------------
# 6. Create 72-hour sequences
# ------------------------------
SEQ_LEN = 72

def create_sequences(dataframe_scaled, seq_len, target_idx):
    data = dataframe_scaled.values.astype(np.float32)
    xs, ys = [], []

    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]            # shape: (72, num_features)
        y = data[i + seq_len, target_idx]  # next-step pm2_5
        xs.append(x)
        ys.append(y)

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN, target_idx)
X_val, y_val     = create_sequences(val_scaled, SEQ_LEN, target_idx)
X_test, y_test   = create_sequences(test_scaled, SEQ_LEN, target_idx)

print("\nSequence shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)


# ------------------------------
# 7. Dataset and loaders
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
# 8. Sinusoidal positional encoding
# ------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)   # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ------------------------------
# 9. Simple Transformer Encoder block
# ------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)     # queries=keys=values=x
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# ------------------------------
# 10. Forecasting model
# ------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        dropout=0.1,
        use_positional_encoding=False,
        seq_len=72
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        # Project raw features to transformer embedding dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        x = self.input_proj(x)   # (B, T, d_model)

        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        # Use final time step representation for forecasting next pm2_5
        last_token = x[:, -1, :]         # (B, d_model)
        out = self.regressor(last_token).squeeze(-1)   # (B,)
        return out


# ------------------------------
# 11. Training utilities
# ------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

    return total_loss / total_count


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_val_loss


def evaluate_test(model, test_loader):
    criterion = nn.MSELoss()
    test_loss = run_epoch(model, test_loader, criterion, optimizer=None)
    return test_loss


# ------------------------------
# 12. Train model WITHOUT positional encoding
# ------------------------------
input_dim = len(available_features)

model_no_pos = TimeSeriesTransformer(
    input_dim=input_dim,
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    num_layers=2,
    dropout=0.1,
    use_positional_encoding=False,
    seq_len=SEQ_LEN
).to(device)

print("\nTraining model WITHOUT positional encoding...\n")
model_no_pos, train_losses_no_pos, val_losses_no_pos, best_val_no_pos = train_model(
    model_no_pos, train_loader, val_loader, epochs=15, lr=1e-3
)

test_loss_no_pos = evaluate_test(model_no_pos, test_loader)

print("\nBest Validation Loss WITHOUT positional encoding:", best_val_no_pos)
print("Test Loss WITHOUT positional encoding:", test_loss_no_pos)


# ------------------------------
# 13. Train model WITH positional encoding
# ------------------------------
model_with_pos = TimeSeriesTransformer(
    input_dim=input_dim,
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    num_layers=2,
    dropout=0.1,
    use_positional_encoding=True,
    seq_len=SEQ_LEN
).to(device)

print("\nTraining model WITH sinusoidal positional encoding...\n")
model_with_pos, train_losses_with_pos, val_losses_with_pos, best_val_with_pos = train_model(
    model_with_pos, train_loader, val_loader, epochs=15, lr=1e-3
)

test_loss_with_pos = evaluate_test(model_with_pos, test_loader)

print("\nBest Validation Loss WITH positional encoding:", best_val_with_pos)
print("Test Loss WITH positional encoding:", test_loss_with_pos)


# ------------------------------
# 14. Compare losses
# ------------------------------
print("\n========== FINAL COMPARISON ==========")
print(f"Validation Loss WITHOUT positional encoding : {best_val_no_pos:.6f}")
print(f"Validation Loss WITH positional encoding    : {best_val_with_pos:.6f}")
print(f"Test Loss WITHOUT positional encoding       : {test_loss_no_pos:.6f}")
print(f"Test Loss WITH positional encoding          : {test_loss_with_pos:.6f}")


# ------------------------------
# 15. Plot training curves
# ------------------------------
plt.figure(figsize=(9, 5))
plt.plot(train_losses_no_pos, label="Train - No Positional Encoding")
plt.plot(val_losses_no_pos, label="Val - No Positional Encoding")
plt.plot(train_losses_with_pos, label="Train - With Positional Encoding")
plt.plot(val_losses_with_pos, label="Val - With Positional Encoding")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Transformer Encoder: With vs Without Positional Encoding")
plt.legend()
plt.grid(True)
plt.savefig('encoder_loss.jpeg')


# ------------------------------
# 16. Plot side-by-side validation curves only
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(val_losses_no_pos, label="Without Positional Encoding")
plt.plot(val_losses_with_pos, label="With Positional Encoding")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.savefig('val-encoder.jpeg')