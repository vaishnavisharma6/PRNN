import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. SETTINGS
# =========================================================

TIME_COL = "date"                                         
FEATURE_COLS = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']    # change to match your file
TARGET_COL = "pm2_5"
SEQ_LEN = 100
HIDDEN_DIM = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 2. LOAD DATASET
# =========================================================
df = pd.read_csv('/Users/vaishnavisharma/prnn/delhi_aqi.csv')

print("Columns in dataset:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 3. KEEP ONLY TIME + CHOSEN FEATURES
# =========================================================
df = df[[TIME_COL] + FEATURE_COLS].copy()


# =========================================================
# 4. SORT CHRONOLOGICALLY
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
# 6. CONVERT FEATURES TO ARRAY
# =========================================================
data_values = df[FEATURE_COLS].values


# =========================================================
# 7. SCALE FEATURES
# =========================================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_values)


# =========================================================
# 8. FIND TARGET COLUMN INDEX
# =========================================================
target_col_idx = FEATURE_COLS.index(TARGET_COL)
print("Target column index:", target_col_idx)
print("Target column name :", FEATURE_COLS[target_col_idx])


# =========================================================
# 9. CREATE 100-LENGTH SEQUENCES
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


X, y = create_sequences(data_scaled, seq_len=SEQ_LEN, target_col_idx=target_col_idx)

print("\nSequence shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# 10. TAKE ONE SAMPLE
# =========================================================
x_sample = torch.tensor(X[0:1], dtype=torch.float32).to(device)   # shape: (1, 100, input_dim)
y_sample = torch.tensor(y[0:1], dtype=torch.float32).to(device)   # shape: (1, 1)

print("\nSample shapes:")
print("x_sample:", x_sample.shape)
print("y_sample:", y_sample.shape)


# =========================================================
# 11. DEFINE VANILLA RNN FROM SCRATCH
# =========================================================
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Wx = nn.Linear(input_dim, hidden_dim)      # input -> hidden
        self.Wh = nn.Linear(hidden_dim, hidden_dim)     # hidden -> hidden
        self.Wy = nn.Linear(hidden_dim, output_dim)     # hidden -> output

    def forward(self, x, return_hidden_states=False):
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        hidden_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(self.Wx(x_t) + self.Wh(h))

            if return_hidden_states:
                h.retain_grad()
                hidden_states.append(h)

        out = self.Wy(h)

        if return_hidden_states:
            return out, hidden_states
        return out


# =========================================================
# 12. CREATE UNTRAINED MODEL
# =========================================================
input_dim = X.shape[2]
model = VanillaRNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=1).to(device)

criterion = nn.MSELoss()


# =========================================================
# 13. FORWARD PASS ON UNTRAINED MODEL
# =========================================================
model.zero_grad()
output, hidden_states = model(x_sample, return_hidden_states=True)
loss = criterion(output, y_sample)

print("\nLoss on untrained model:", loss.item())


# =========================================================
# 14. BACKWARD PASS
# =========================================================
loss.backward()


# =========================================================
# 15. EXTRACT GRADIENT NORMS AT t=0, 50, 99
# =========================================================
grad_t0 = hidden_states[0].grad.norm().item()
grad_t50 = hidden_states[49].grad.norm().item()
grad_t99 = hidden_states[99].grad.norm().item()

print("\nGradient norm at t=0  :", grad_t0)
print("Gradient norm at t=50 :", grad_t50)
print("Gradient norm at t=99 :", grad_t99)


# =========================================================
# 16. PLOT GRADIENT NORM ACROSS ALL 100 TIME STEPS
# =========================================================
grad_norms = [h.grad.norm().item() for h in hidden_states]

plt.figure(figsize=(8, 5))
plt.plot(range(100), grad_norms)
plt.xlabel("Time step")
plt.ylabel("Gradient norm")
plt.title("Gradient norm of loss w.r.t. hidden state")
plt.grid(True)
plt.savefig('grad-t.jpeg')