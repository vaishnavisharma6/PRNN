import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. CONFIG
SEQ_LEN = 72
BATCH_SIZE = 64
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# =========================================================
# 2. LOAD DATA
# =========================================================
df = pd.read_csv('/Users/vaishnavisharma/prnn/delhi_aqi.csv')

print("Columns in dataset:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 3. FIND TIME COLUMN
# =========================================================
possible_time_cols = ["date"]

time_col = None
for col in possible_time_cols:
    if col in df.columns:
        time_col = col
        break

if time_col is None:
    raise ValueError(
        "Time column not found automatically. Please set time_col manually."
    )

print("\nUsing time column:", time_col)

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col])
df = df.sort_values(time_col).reset_index(drop=True)


# =========================================================
# 4. SELECT FEATURES
# =========================================================
# Adjust this list based on your dataset columns
candidate_features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

available_features = [col for col in candidate_features if col in df.columns]

if "pm2_5" not in available_features:
    raise ValueError("PM2.5 column is required in the dataset.")


print("\nUsing features:")
print(available_features)

df = df[[time_col] + available_features].copy()


# =========================================================
# 5. HANDLE MISSING VALUES
# =========================================================
df[available_features] = df[available_features].replace([np.inf, -np.inf], np.nan)
df[available_features] = df[available_features].ffill().bfill()
df = df.dropna(subset=available_features).reset_index(drop=True)

print("\nRows after cleaning:", len(df))


# =========================================================
# 6. CHRONOLOGICAL 70 / 15 / 15 SPLIT
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
# Fit scaler ONLY on train data
# =========================================================
train_values = train_df[available_features].values
val_values = val_df[available_features].values
test_values = test_df[available_features].values

print(df[available_features[5]])
feature_scaler = StandardScaler()
train_scaled = feature_scaler.fit_transform(train_values)
val_scaled = feature_scaler.transform(val_values)
test_scaled = feature_scaler.transform(test_values)

# Separate scaler for PM2.5 target for inverse transform later
target_scaler = StandardScaler()
target_scaler.fit(train_df[["pm2_5"]].values)


# =========================================================
# 8. CREATE SEQUENCES
# Input: previous 72 steps
# Output: next PM2.5
# =========================================================
def create_sequences(data_array, seq_len=72, target_col_idx=5):
    X, y = [], []

    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i + seq_len])
        y.append(data_array[i + seq_len, target_col_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


X_train, y_train = create_sequences(train_scaled, seq_len=SEQ_LEN, target_col_idx=5)
X_val, y_val = create_sequences(val_scaled, seq_len=SEQ_LEN, target_col_idx=5)
X_test, y_test = create_sequences(test_scaled, seq_len=SEQ_LEN, target_col_idx=5)

print("\nSequence shapes:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val  :", X_val.shape,   " y_val  :", y_val.shape)
print("X_test :", X_test.shape,  " y_test :", y_test.shape)


# =========================================================
# 9. DATASET CLASS
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
test_dataset = AirQualityDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================================================
# 10. VANILLA RNN FROM SCRATCH
# =========================================================
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Wx = nn.Linear(input_dim, hidden_dim)      # input -> hidden
        self.Wh = nn.Linear(hidden_dim, hidden_dim)     # hidden -> hidden
        self.Wy = nn.Linear(hidden_dim, output_dim)     # hidden -> output

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]   # shape: (batch_size, input_dim)
            h = torch.tanh(self.Wx(x_t) + self.Wh(h))

        out = self.Wy(h)       # shape: (batch_size, 1)
        return out


input_dim = X_train.shape[2]
model = VanillaRNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=1).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =========================================================
# 11. TRAINING AND EVALUATION FUNCTIONS
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


# =========================================================
# 12. TRAIN MODEL
# =========================================================
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")


# =========================================================
# 13. TEST EVALUATION
# =========================================================
test_loss = evaluate(model, test_loader, criterion, DEVICE)
print(f"\nTest Loss: {test_loss:.6f}")


# =========================================================
# 14. PLOT TRAIN / VAL LOSS
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Vanilla RNN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig('train-val-rnn.jpeg')


# =========================================================
# 15. PREDICT ON TEST SET
# =========================================================
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).cpu().numpy()

        all_preds.append(preds)
        all_true.append(y_batch.numpy())

all_preds = np.vstack(all_preds)
all_true = np.vstack(all_true)

# Convert back to original PM2.5 scale
preds_pm25 = target_scaler.inverse_transform(all_preds)
true_pm25 = target_scaler.inverse_transform(all_true)


# =========================================================
# 16. PLOT TEST PREDICTIONS
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(true_pm25[:200], label="True PM2.5")
plt.plot(preds_pm25[:200], label="Predicted PM2.5")
plt.xlabel("Time Step")
plt.ylabel("PM2.5")
plt.title("True vs Predicted PM2.5 on Test Set")
plt.legend()
plt.grid(True)
plt.savefig('test-rnn.jpeg')


#second part
