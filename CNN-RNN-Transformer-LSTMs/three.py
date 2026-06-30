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
CSV_PATH = "/Users/vaishnavisharma/prnn/delhi_aqi.csv"   
TARGET_COL = "pm2_5"

INPUT_SEQ_LEN = 72     # past 72 hours
OUTPUT_SEQ_LEN = 24    # next 24 hours

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5

# ============================================================
# 3. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = [c.strip().lower() for c in df.columns]
TARGET_COL = TARGET_COL.lower()

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.\nColumns: {df.columns.tolist()}")

# Use only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COL not in numeric_cols:
    raise ValueError(f"Target column '{TARGET_COL}' is not numeric.")

feature_cols = numeric_cols.copy()

# Drop rows with missing values
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
# 5. STANDARDIZE USING TRAIN ONLY
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

# We will need these later to convert predictions back to original units
target_mean = train_mean[TARGET_COL]
target_std = train_std[TARGET_COL]

# ============================================================
# 6. DATASET FOR SEQ2SEQ
#
# Input:
#   past 72 hours of all features
#
# Target:
#   next 24 hours of PM2.5 only
#
# Example:
#   x = data[t : t+72]
#   y = pm2_5[t+72 : t+72+24]
# ============================================================
class AirQualitySeq2SeqDataset(Dataset):
    def __init__(self, scaled_df, input_seq_len, output_seq_len, target_col_index):
        self.data = scaled_df.values.astype(np.float32)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.target_col_index = target_col_index

    def __len__(self):
        return len(self.data) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_seq_len]  # [72, num_features]

        y = self.data[
            idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len,
            self.target_col_index
        ]  # [24]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = AirQualitySeq2SeqDataset(train_scaled, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, target_col_index)
val_dataset = AirQualitySeq2SeqDataset(val_scaled, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, target_col_index)
test_dataset = AirQualitySeq2SeqDataset(test_scaled, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, target_col_index)

print("\nDataset lengths:")
print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 7. ENCODER
#
# The encoder reads the past 72-hour input sequence and
# compresses it into a final hidden state and cell state.
# ============================================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        # x shape: [batch, 72, input_size]
        outputs, (hidden, cell) = self.lstm(x)

        # outputs: [batch, 72, hidden_size]
        # hidden : [1, batch, hidden_size]
        # cell   : [1, batch, hidden_size]
        return hidden, cell

# ============================================================
# 8. DECODER
#
# The decoder predicts one future step at a time.
# Input to decoder at each step:
#   previous PM2.5 value (scalar)
#
# Output:
#   next PM2.5 value
# ============================================================
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        # Decoder input at each step is a single PM2.5 value
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: [batch, 1, 1]
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        # output shape: [batch, 1, hidden_size]
        pred = self.fc(output[:, 0, :])   # [batch, 1]

        return pred, hidden, cell

# ============================================================
# 9. SEQ2SEQ MODEL
#
# Encoder:
#   reads past 72 hours
#
# Decoder:
#   predicts 24 future PM2.5 values autoregressively
#
# Teacher forcing:
#   sometimes feed the true previous target
#   sometimes feed the model's own previous prediction
# ============================================================
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_seq_len):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size=1)
        self.output_seq_len = output_seq_len

    def forward(self, src, target=None, teacher_forcing_ratio=0.0):
        """
        src    : [batch, 72, input_size]
        target : [batch, 24] or None
        """
        batch_size = src.size(0)

        # Store all decoder predictions
        outputs = torch.zeros(batch_size, self.output_seq_len, device=src.device)

        # Encoder processes past 72-hour input
        hidden, cell = self.encoder(src)

        # First decoder input:
        # use the last PM2.5 value from the input sequence
        decoder_input = src[:, -1, target_col_index].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]

        for t in range(self.output_seq_len):
            pred, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = pred.squeeze(1)

            # Decide whether to use teacher forcing
            if target is not None and np.random.rand() < teacher_forcing_ratio:
                next_input = target[:, t].unsqueeze(1).unsqueeze(2)  # true value
            else:
                next_input = pred.unsqueeze(1)  # model prediction, shape [batch, 1, 1]

            decoder_input = next_input

        return outputs

# ============================================================
# 10. HELPER FUNCTIONS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, teacher_forcing_ratio):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        preds = model(x_batch, target=y_batch, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch, target=None, teacher_forcing_ratio=0.0)
            loss = criterion(preds, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def inverse_transform_target(seq_scaled, mean, std):
    return seq_scaled * std + mean

# ============================================================
# 11. BUILD MODEL
# ============================================================
model = Seq2SeqLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_seq_len=OUTPUT_SEQ_LEN
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# 12. TRAINING LOOP
# ============================================================
train_losses = []
val_losses = []
epoch_times = []

print("\nTraining Seq2Seq LSTM...")

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        teacher_forcing_ratio=TEACHER_FORCING_RATIO
    )

    val_loss = evaluate(model, val_loader, criterion, device)

    end_time = time.time()
    epoch_time = end_time - start_time

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_times.append(epoch_time)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f} | "
        f"Time: {epoch_time:.2f} sec"
    )

# ============================================================
# 13. TEST SET EVALUATION
# ============================================================
test_loss = evaluate(model, test_loader, criterion, device)

print("\nFinal Validation MSE:", val_losses[-1])
print("Test MSE:", test_loss)

# ============================================================
# 14. PLOT TRAIN / VAL LOSS
# ============================================================
epochs = np.arange(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label="Train MSE")
plt.plot(epochs, val_losses, marker='o', label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Seq2Seq LSTM Training")
plt.legend()
plt.tight_layout()
plt.savefig('lstm-jpeg')

# ============================================================
# 15. PLOT ONE TEST EXAMPLE
#
# We take one example from test set, predict the next 24 hours,
# and compare predicted vs true PM2.5 sequence.
# ============================================================
model.eval()

x_example, y_true = test_dataset[0]
x_example = x_example.unsqueeze(0).to(device)  # [1, 72, input_size]

with torch.no_grad():
    y_pred = model(x_example, target=None, teacher_forcing_ratio=0.0)

# Convert to numpy
y_true = y_true.cpu().numpy()              # [24]
y_pred = y_pred.squeeze(0).cpu().numpy()  # [24]

# Convert back to original PM2.5 units
y_true_original = inverse_transform_target(y_true, target_mean, target_std)
y_pred_original = inverse_transform_target(y_pred, target_mean, target_std)

future_hours = np.arange(1, OUTPUT_SEQ_LEN + 1)

plt.figure(figsize=(10, 5))
plt.plot(future_hours, y_true_original, marker='o', label="True PM2.5")
plt.plot(future_hours, y_pred_original, marker='o', label="Predicted PM2.5")
plt.xlabel("Future hour")
plt.ylabel("PM2.5")
plt.title("24-hour Forecast: True vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig('true-pred.jpeg')

# ============================================================
# 16. OPTIONAL: PRINT PREDICTION ARRAYS
# ============================================================
print("\nTrue 24-hour PM2.5 sequence:")
print(y_true_original)

print("\nPredicted 24-hour PM2.5 sequence:")
print(y_pred_original)