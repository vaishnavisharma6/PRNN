import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. REPRODUCIBILITY + DEVICE
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. USER SETTINGS
# ============================================================
CSV_PATH = "/Users/vaishnavisharma/prnn/delhi_aqi.csv"   # <-- change this
TARGET_COL = "pm2_5"
SEQ_LEN = 100
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

# ============================================================
# 3. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

df.columns = [c.strip().lower() for c in df.columns]
TARGET_COL = TARGET_COL.lower()

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.\nColumns: {df.columns.tolist()}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COL not in numeric_cols:
    raise ValueError(f"Target column '{TARGET_COL}' is not numeric.")

feature_cols = numeric_cols.copy()

df = df[feature_cols].dropna().reset_index(drop=True)

print("Using features:")
print(feature_cols)
print("Number of rows after cleanup:", len(df))

# ============================================================
# 4. CHRONOLOGICAL SPLIT: 70 / 15 / 15
# ============================================================
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

# ============================================================
# 5. STANDARDIZATION USING TRAIN ONLY
# ============================================================
train_mean = train_df.mean()
train_std = train_df.std().replace(0, 1.0)

train_scaled = (train_df - train_mean) / train_std
val_scaled = (val_df - train_mean) / train_std
test_scaled = (test_df - train_mean) / train_std

target_col_index = feature_cols.index(TARGET_COL)
input_size = len(feature_cols)

print("\nTarget column index:", target_col_index)
print("Input size:", input_size)

# ============================================================
# 6. DATASET
#    Input  : past 100 time steps
#    Output : PM2.5 at final step of that window
# ============================================================
class AirQualitySeqDataset(Dataset):
    def __init__(self, scaled_df, seq_len, target_col_index):
        self.data = scaled_df.values.astype(np.float32)
        self.seq_len = seq_len
        self.target_col_index = target_col_index

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]   # [seq_len, num_features]
        y = self.data[idx + self.seq_len - 1, self.target_col_index]  # scalar
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = AirQualitySeqDataset(train_scaled, SEQ_LEN, target_col_index)
val_dataset = AirQualitySeqDataset(val_scaled, SEQ_LEN, target_col_index)
test_dataset = AirQualitySeqDataset(test_scaled, SEQ_LEN, target_col_index)

print("\nDataset lengths:")
print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 7. MODELS
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, (hn, cn) = self.lstm(x)
        # use final time step output
        y = self.fc(out[:, -1, :]).squeeze(-1)
        return y


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, hn = self.gru(x)
        # use final time step output
        y = self.fc(out[:, -1, :]).squeeze(-1)
        return y

# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / total_samples


def train_model(model, train_loader, val_loader, num_epochs, lr, device, model_name="Model"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    epoch_times = []

    print(f"\nTraining {model_name}...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_time = end_time - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f} | "
            f"Time: {epoch_time:.2f} sec"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "final_val_mse": val_losses[-1],
        "avg_epoch_time": float(np.mean(epoch_times))
    }

# ============================================================
# 9. BUILD MODELS
# ============================================================
lstm_model = LSTMRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE).to(device)
gru_model = GRURegressor(input_size=input_size, hidden_size=HIDDEN_SIZE).to(device)

lstm_params = count_trainable_parameters(lstm_model)
gru_params = count_trainable_parameters(gru_model)

print("\nTrainable parameter counts:")
print("LSTM:", lstm_params)
print("GRU :", gru_params)

# ============================================================
# 10. TRAIN BOTH MODELS
# ============================================================
lstm_results = train_model(
    model=lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    device=device,
    model_name="LSTM"
)

gru_results = train_model(
    model=gru_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    device=device,
    model_name="GRU"
)

# ============================================================
# 11. FINAL COMPARISON
# ============================================================
print("\n================ FINAL COMPARISON ================")
print(f"LSTM Parameters         : {lstm_params}")
print(f"GRU Parameters          : {gru_params}")
print()
print(f"LSTM Avg Epoch Time     : {lstm_results['avg_epoch_time']:.4f} sec")
print(f"GRU Avg Epoch Time      : {gru_results['avg_epoch_time']:.4f} sec")
print()
print(f"LSTM Final Val MSE      : {lstm_results['final_val_mse']:.6f}")
print(f"GRU Final Val MSE       : {gru_results['final_val_mse']:.6f}")
print("==================================================")

# ============================================================
# 12. PLOT VALIDATION MSE CURVES
# ============================================================
epochs = np.arange(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, lstm_results["val_losses"], marker='o', label="LSTM Val MSE")
plt.plot(epochs, gru_results["val_losses"], marker='o', label="GRU Val MSE")
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Validation MSE: LSTM vs GRU")
plt.legend()
plt.tight_layout()
plt.savefig('lstm-gru.jpeg')

# ============================================================
# 13. PLOT EPOCH TIME COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(epochs, lstm_results["epoch_times"], marker='o', label="LSTM Epoch Time")
plt.plot(epochs, gru_results["epoch_times"], marker='o', label="GRU Epoch Time")
plt.xlabel("Epoch")
plt.ylabel("Time per epoch (seconds)")
plt.title("Training Time per Epoch: LSTM vs GRU")
plt.legend()
plt.tight_layout()
plt.savefig('time-lstm-gru.jpeg')

# ============================================================
# 14. OPTIONAL TEST EVALUATION
# ============================================================
criterion = nn.MSELoss()

lstm_test_mse = evaluate(lstm_model, test_loader, criterion, device)
gru_test_mse = evaluate(gru_model, test_loader, criterion, device)

print("\n================ TEST SET RESULTS ================")
print(f"LSTM Test MSE : {lstm_test_mse:.6f}")
print(f"GRU Test MSE  : {gru_test_mse:.6f}")
print("==================================================")