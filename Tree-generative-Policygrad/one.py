import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# 1. Load Delhi AQI dataset


CSV_PATH = "/Users/vaishnavisharma/prnn-3/delhi_aqi.csv"   

df = pd.read_csv(CSV_PATH)

print("Original columns:")
print(df.columns)

for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        df = df.drop(columns=[col])

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

#missing values
df = df.ffill().bfill()

print("Numeric columns used:")
print(df.columns)

data = df.values


# =========================
# 2. Standardize data


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# =========================
# 3. Create 72-hour windows


def create_sequences(data, window_size=72):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)


WINDOW_SIZE = 72

X_seq = create_sequences(data_scaled, WINDOW_SIZE)

print("Sequence shape:", X_seq.shape)
# shape = (num_samples, 72, num_features)

X_flat = X_seq.reshape(X_seq.shape[0], -1)

print("Flattened shape:", X_flat.shape)
# shape = (num_samples, 72 * num_features)


# =========================
# 4. Train-test split


X_train, X_test = train_test_split(
    X_flat,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# =========================
# 5. PCA using SVD


mean_train = np.mean(X_train, axis=0)
X_train_centered = X_train - mean_train
X_test_centered = X_test - mean_train

U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)

explained_variance = (S ** 2) / (X_train.shape[0] - 1)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

cumulative_variance = np.cumsum(explained_variance_ratio)

k = np.argmax(cumulative_variance >= 0.95) + 1

print("\nNumber of PCA components for 95% variance:", k)


# PCA reconstruction
Vt_k = Vt[:k, :]

X_test_pca_encoded = X_test_centered @ Vt_k.T
X_test_pca_reconstructed = X_test_pca_encoded @ Vt_k + mean_train

pca_mse = mean_squared_error(X_test, X_test_pca_reconstructed)

print("PCA Reconstruction MSE:", pca_mse)


# =========================
# 6. Linear Autoencoder


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()

        self.encoder = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.decoder = nn.Linear(bottleneck_dim, input_dim, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


input_dim = X_train.shape[1]

model = LinearAutoencoder(input_dim, k)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_train_tensor = torch.tensor(X_train_centered, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_centered, dtype=torch.float32)

EPOCHS = 500
BATCH_SIZE = 64

dataset = torch.utils.data.TensorDataset(X_train_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        x_batch = batch[0]

        optimizer.zero_grad()
        x_recon = model(x_batch)
        loss = criterion(x_recon, x_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(loader):.6f}")


# AE reconstruction
model.eval()

with torch.no_grad():
    X_test_ae_reconstructed_centered = model(X_test_tensor).numpy()

X_test_ae_reconstructed = X_test_ae_reconstructed_centered + mean_train

ae_mse = mean_squared_error(X_test, X_test_ae_reconstructed)

print("\nLinear Autoencoder Reconstruction MSE:", ae_mse)


# =========================
# 7. Final comparison


print("\n========== Final Results ==========")
print("PCA components for 95% variance:", k)
print("PCA Reconstruction MSE:", pca_mse)
print("Linear AE Reconstruction MSE:", ae_mse)

if abs(pca_mse - ae_mse) / pca_mse < 0.1:
    print("Conclusion: Linear AE and PCA give very similar reconstruction MSE.")
    print("This empirically shows that the linear AE learns approximately the same subspace as PCA.")
else:
    print("Conclusion: AE MSE is not very close to PCA MSE. Try training longer or reducing learning rate.")