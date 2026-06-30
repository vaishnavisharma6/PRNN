# ============================================================
# PHASE 4.10 — LSTM Gradient Rescue
# Assignment-correct script
#
# What this script does:
# 1. Loads Delhi air-quality data
# 2. Uses a chronological 70/15/15 split
# 3. Standardizes using TRAIN statistics only
# 4. Creates 100-step sequences
# 5. Builds:
#       - Vanilla RNN from scratch
#       - LSTM using nn.LSTMCell unrolled manually
# 6. Uses UNTRAINED models
# 7. Computes loss and calls backward()
# 8. Extracts gradient magnitudes at t=0, t=50, t=100
# 9. Plots Vanilla RNN vs LSTM gradient magnitudes
#
# This is written specifically to satisfy the assignment.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
CSV_PATH = "/Users/vaishnavisharma/prnn/delhi_aqi.csv"   # <-- change this to your actual csv path
TARGET_COL = "pm2_5"                 # PM2.5 column
SEQ_LEN = 100                        # required by assignment
HIDDEN_SIZE = 64

# ============================================================
# 3. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# Make column names clean and lowercase
df.columns = [c.strip().lower() for c in df.columns]
TARGET_COL = TARGET_COL.lower()

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.\nColumns: {df.columns.tolist()}")

# Keep numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COL not in numeric_cols:
    raise ValueError(f"Target column '{TARGET_COL}' is not numeric.")

# Use all numeric columns as features
feature_cols = numeric_cols.copy()

# Drop missing rows only for selected columns
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
# 5. STANDARDIZATION USING TRAIN SET ONLY
# ============================================================
train_mean = train_df.mean()
train_std = train_df.std()

# Avoid divide-by-zero if any feature is constant
train_std = train_std.replace(0, 1.0)

train_scaled = (train_df - train_mean) / train_std
val_scaled = (val_df - train_mean) / train_std
test_scaled = (test_df - train_mean) / train_std

target_col_index = feature_cols.index(TARGET_COL)
input_size = len(feature_cols)

print("\nTarget column index:", target_col_index)
print("Input size:", input_size)

# ============================================================
# 6. DATASET
#    Input  : 100-step sequence
#    Target : PM2.5 at final time step of the window
# ============================================================
class AirQualitySeqDataset(Dataset):
    def __init__(self, scaled_df, seq_len, target_col_index):
        self.data = scaled_df.values.astype(np.float32)
        self.seq_len = seq_len
        self.target_col_index = target_col_index

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]  # [seq_len, num_features]
        y = self.data[idx + self.seq_len - 1, self.target_col_index]  # scalar target at last step

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

train_dataset = AirQualitySeqDataset(train_scaled, SEQ_LEN, target_col_index)
val_dataset = AirQualitySeqDataset(val_scaled, SEQ_LEN, target_col_index)
test_dataset = AirQualitySeqDataset(test_scaled, SEQ_LEN, target_col_index)

print("\nDataset lengths:")
print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))

if len(test_dataset) == 0:
    raise ValueError("Test dataset has zero sequences. Need at least 100 rows in test split.")

# ============================================================
# 7. VANILLA RNN FROM SCRATCH
#    Implemented using only:
#    - nn.Linear
#    - tanh
#    - Python for-loop over time
# ============================================================
class VanillaRNNFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2y = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h0=None, return_all_hidden=False):
        # x shape: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.shape

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0

        hidden_states = [h_t]  # h_0

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.tanh(self.i2h(x_t) + self.h2h(h_t))
            hidden_states.append(h_t)  # h_1, ..., h_100

        y = self.h2y(h_t).squeeze(-1)

        if return_all_hidden:
            return y, hidden_states
        return y

# ============================================================
# 8. LSTM MODEL FOR GRADIENT INSPECTION
#    We use nn.LSTMCell and manually unroll it so that
#    hidden states at every time step are directly accessible.
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None, return_all_hidden=False):
        # x shape: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.shape

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0

        if c0 is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            c_t = c0

        hidden_states = [h_t]  # h_0
        cell_states = [c_t]    # c_0

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            hidden_states.append(h_t)  # h_1, ..., h_100
            cell_states.append(c_t)

        y = self.fc(h_t).squeeze(-1)

        if return_all_hidden:
            return y, hidden_states, cell_states
        return y

# ============================================================
# 9. CHOOSE ONE 100-STEP TEST SAMPLE
#    Assignment asks for passing a 100-step sequence through the
#    untrained network. Using one sample is appropriate.
# ============================================================
x_sample, y_sample = test_dataset[0]
x_sample = x_sample.unsqueeze(0).to(device)  # [1, 100, input_size]
y_sample = y_sample.unsqueeze(0).to(device)  # [1]

print("\nSingle sample shapes:")
print("x_sample:", x_sample.shape)
print("y_sample:", y_sample.shape)

# ============================================================
# 10. GRADIENT EXTRACTION FOR VANILLA RNN
# ============================================================
def get_vanilla_rnn_gradients(model, x, y):
    model.zero_grad()

    batch_size = x.size(0)

    # h0 corresponds to t = 0
    h0 = torch.zeros(batch_size, model.hidden_size, device=x.device, requires_grad=True)

    pred, hidden_states = model(x, h0=h0, return_all_hidden=True)

    # hidden_states has length 101:
    # hidden_states[0]   = h0     -> t = 0
    # hidden_states[50]  = h50    -> t = 50
    # hidden_states[100] = h100   -> t = 100
    hidden_states[0].retain_grad()
    hidden_states[50].retain_grad()
    hidden_states[100].retain_grad()

    loss = nn.MSELoss()(pred, y)
    loss.backward()

    grad_t0 = hidden_states[0].grad
    grad_t50 = hidden_states[50].grad
    grad_t100 = hidden_states[100].grad

    return {
        "loss": loss.item(),
        "t0": grad_t0.norm().item() if grad_t0 is not None else 0.0,
        "t50": grad_t50.norm().item() if grad_t50 is not None else 0.0,
        "t100": grad_t100.norm().item() if grad_t100 is not None else 0.0
    }

# ============================================================
# 11. GRADIENT EXTRACTION FOR LSTM
# ============================================================
def get_lstm_gradients(model, x, y):
    model.zero_grad()

    batch_size = x.size(0)

    # h0 corresponds to t = 0
    h0 = torch.zeros(batch_size, model.hidden_size, device=x.device, requires_grad=True)
    c0 = torch.zeros(batch_size, model.hidden_size, device=x.device, requires_grad=True)

    pred, hidden_states, cell_states = model(x, h0=h0, c0=c0, return_all_hidden=True)

    # hidden_states indexing:
    # hidden_states[0]   = h0     -> t = 0
    # hidden_states[50]  = h50    -> t = 50
    # hidden_states[100] = h100   -> t = 100
    hidden_states[0].retain_grad()
    hidden_states[50].retain_grad()
    hidden_states[100].retain_grad()

    loss = nn.MSELoss()(pred, y)
    loss.backward()

    grad_t0 = hidden_states[0].grad
    grad_t50 = hidden_states[50].grad
    grad_t100 = hidden_states[100].grad

    return {
        "loss": loss.item(),
        "t0": grad_t0.norm().item() if grad_t0 is not None else 0.0,
        "t50": grad_t50.norm().item() if grad_t50 is not None else 0.0,
        "t100": grad_t100.norm().item() if grad_t100 is not None else 0.0
    }

# ============================================================
# 12. BUILD UNTRAINED MODELS
# ============================================================
vanilla_rnn = VanillaRNNFromScratch(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=1
).to(device)

lstm_model = LSTMRegressor(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=1
).to(device)

# ============================================================
# 13. RUN EXPERIMENT
# ============================================================
rnn_grads = get_vanilla_rnn_gradients(vanilla_rnn, x_sample, y_sample)
lstm_grads = get_lstm_gradients(lstm_model, x_sample, y_sample)

print("\nVanilla RNN gradients:")
print(rnn_grads)

print("\nLSTM gradients:")
print(lstm_grads)

# ============================================================
# 14. PLOT BAR CHART
# ============================================================
time_labels = ["t=0", "t=50", "t=100"]

rnn_vals = [rnn_grads["t0"], rnn_grads["t50"], rnn_grads["t100"]]
lstm_vals = [lstm_grads["t0"], lstm_grads["t50"], lstm_grads["t100"]]

xpos = np.arange(len(time_labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(xpos - width/2, rnn_vals, width, label="Vanilla RNN")
plt.bar(xpos + width/2, lstm_vals, width, label="LSTM")
plt.xticks(xpos, time_labels)
plt.ylabel("Gradient magnitude (L2 norm)")
plt.title("Gradient comparison at t=0, 50, 100")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 15. OPTIONAL LOG-SCALE PLOT
#    Useful because vanishing gradients can differ by orders
#    of magnitude. This often makes the comparison clearer.
# ============================================================
eps = 1e-12
plt.figure(figsize=(8, 5))
plt.bar(xpos - width/2, np.array(rnn_vals) + eps, width, label="Vanilla RNN")
plt.bar(xpos + width/2, np.array(lstm_vals) + eps, width, label="LSTM")
plt.xticks(xpos, time_labels)
plt.yscale("log")
plt.ylabel("Gradient magnitude (log scale)")
plt.title("Gradient comparison at t=0, 50, 100 (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig('lstm-rnn.jpeg')

# ============================================================
# 16. CLEAN SUMMARY
# ============================================================
print("\n================ FINAL SUMMARY ================")
print(f"Vanilla RNN loss : {rnn_grads['loss']:.6f}")
print(f"LSTM loss        : {lstm_grads['loss']:.6f}")
print()
print(f"Vanilla RNN -> t=0:   {rnn_grads['t0']:.6e}")
print(f"Vanilla RNN -> t=50:  {rnn_grads['t50']:.6e}")
print(f"Vanilla RNN -> t=100: {rnn_grads['t100']:.6e}")
print()
print(f"LSTM -> t=0:   {lstm_grads['t0']:.6e}")
print(f"LSTM -> t=50:  {lstm_grads['t50']:.6e}")
print(f"LSTM -> t=100: {lstm_grads['t100']:.6e}")
print("==============================================")