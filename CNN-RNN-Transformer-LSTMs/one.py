import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, average_precision_score

# read data from kaggle
df = pd.read_csv('/Users/vaishnavisharma/prnn/delhi_aqi.csv')

df = df.drop(columns=['date'])
features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

epochs = 1300
data = df[features].values
print(data)

# create past 72 hours data and t+24 pm2.5 labels
X = []
Y = []
Y_class = []

past = 72
future = 24

for t in range(past, len(data) - future):
    X.append(data[t-past:t])
    Y.append(data[t+future][5])
    Y_class.append(1 if data[t+future][5] > 200 else 0)

X = np.array(X)
Y = np.array(Y)
Y_class = np.array(Y_class)

N = X.shape[0]
train = int(0.7 * N)
val = int(0.85 * N)

X_train = X[:train]
X_val = X[train:val]
X_test = X[val:]

Y_train = Y[:train]
Y_val = Y[train:val]
Y_test = Y[val:]

c_train = Y_class[:train]
c_val = Y_class[train:val]
c_test = Y_class[val:]

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Y_val = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

c_train = torch.tensor(c_train, dtype=torch.float32).view(-1, 1)
c_val = torch.tensor(c_val, dtype=torch.float32).view(-1, 1)
c_test = torch.tensor(c_test, dtype=torch.float32).view(-1, 1)

# architecture
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

train_losses = []
val_losses = []

model = MLP(input_dim=576)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

# train model
for epoch in range(300):
    model.train()
    preds = model(X_train)
    loss = criterion(preds, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, Y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, Y_test)

print("Part 1.1 Test MSE:", test_loss.item())
print("Part 1.1 Test RMSE:", torch.sqrt(test_loss).item())

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.savefig('loss.jpeg')
plt.show()


# re-initialization with sigmoid
class mlpsig(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.a1 = nn.Sigmoid()
        self.fc2 = nn.Linear(256, 128)
        self.a2 = nn.Sigmoid()
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        x = self.a2(x)
        x = self.fc3(x)
        return x

model = mlpsig(input_dim=576)

train_losses = []
val_losses = []

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

# store gradient norms
grad_fc1 = []
grad_fc2 = []
grad_fc3 = []

def hook_fc1(grad):
    grad_fc1.append(grad.norm(2).item())

def hook_fc2(grad):
    grad_fc2.append(grad.norm(2).item())

def hook_fc3(grad):
    grad_fc3.append(grad.norm(2).item())

h1 = model.fc1.weight.register_hook(hook_fc1)
h2 = model.fc2.weight.register_hook(hook_fc2)
h3 = model.fc3.weight.register_hook(hook_fc3)

for epoch in range(300):
    model.train()
    preds = model(X_train)
    loss = criterion(preds, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, Y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

h1.remove()
h2.remove()
h3.remove()

model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, Y_test)

print("Part 1.2 Test MSE:", test_loss.item())
print("Part 1.2 Test RMSE:", torch.sqrt(test_loss).item())

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve with Sigmoid')
plt.legend()
plt.savefig('loss_sig.jpeg')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(grad_fc1, label='First Linear Layer')
plt.plot(grad_fc2, label='Second Linear Layer')
plt.plot(grad_fc3, label='Third Linear Layer')
plt.xlabel('Backward step')
plt.ylabel('L2 norm of gradient')
plt.title('Gradient Flow in Sigmoid MLP')
plt.legend()
plt.savefig('gradient_flow_sigmoid.jpeg')
plt.show()


# part 1.3
class cmlp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

train_losses = []
val_losses = []

model = cmlp(input_dim=576)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.BCELoss()

for epoch in range(300):
    model.train()
    preds = model(X_train)
    loss = criterion(preds, c_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, c_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, c_test)

print("BCELoss Test Loss:", test_loss.item())

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BCELoss Learning Curve')
plt.legend()
plt.savefig('loss_bce.jpeg')
plt.show()

with torch.no_grad():
    test_probs_bce = model(X_test).cpu().numpy().ravel()


class cmlp_logits(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

train_losses = []
val_losses = []

num_pos = c_train.sum().item()
num_neg = len(c_train) - num_pos
pos_weight_value = num_neg / num_pos
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

print("Positive samples:", num_pos)
print("Negative samples:", num_neg)
print("pos_weight:", pos_weight.item())

model_logits = cmlp_logits(input_dim=576)
optimizer = torch.optim.Adam(model_logits.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(300):
    model_logits.train()
    logits = model_logits(X_train)
    loss = criterion(logits, c_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model_logits.eval()
    with torch.no_grad():
        val_logits = model_logits(X_val)
        val_loss = criterion(val_logits, c_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

model_logits.eval()
with torch.no_grad():
    test_logits = model_logits(X_test)
    test_loss = criterion(test_logits, c_test)
    test_probs_logits = torch.sigmoid(test_logits).cpu().numpy().ravel()

print("BCEWithLogitsLoss Test Loss:", test_loss.item())

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BCEWithLogitsLoss Learning Curve')
plt.legend()
plt.savefig('loss_bce_logits.jpeg')
plt.show()

# Precision-Recall curves
y_true = c_test.cpu().numpy().ravel()

precision_bce, recall_bce, _ = precision_recall_curve(y_true, test_probs_bce)
precision_logits, recall_logits, _ = precision_recall_curve(y_true, test_probs_logits)

ap_bce = average_precision_score(y_true, test_probs_bce)
ap_logits = average_precision_score(y_true, test_probs_logits)

print("Average Precision for BCELoss model:", ap_bce)
print("Average Precision for BCEWithLogitsLoss model:", ap_logits)

plt.figure(figsize=(10, 6))
plt.plot(recall_bce, precision_bce, label=f'BCELoss (AP = {ap_bce:.4f})')
plt.plot(recall_logits, precision_logits, label=f'BCEWithLogitsLoss + pos_weight (AP = {ap_logits:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.savefig('pr_curve.jpeg')
plt.show()