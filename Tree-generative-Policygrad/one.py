import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from sklearn.preprocessing import StandardScaler



# 1. Load data
csv_path = "/Users/vaishnavisharma/prnn-3/delhi_aqi.csv" 

df = pd.read_csv(csv_path)

# Drop date/time columns if present
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        df = df.drop(columns=[col])

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Fill missing values
df = df.fillna(method="ffill").fillna(method="bfill")

print("Columns:", df.columns.tolist())



# 2. Choose PM2.5 column

pm25_col = None
for col in df.columns:
    if "pm2_5" in col.lower():
        pm25_col = col
        break

if pm25_col is None:
    raise ValueError("No PM2.5 column found. Please rename your PM2.5 column or set pm25_col manually.")

print("Using PM2.5 column:", pm25_col)



# 3. Normalize features
features = df.values.astype(np.float32)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features).astype(np.float32)

pm25_values = df[pm25_col].values.astype(np.float32)



# 4. Create RL environment
class AirQualityEnv:
    def __init__(self, features_scaled, pm25_values, window_size=24, episode_length=200):
        self.features_scaled = features_scaled
        self.pm25_values = pm25_values
        self.window_size = window_size
        self.episode_length = episode_length

        self.num_steps = len(features_scaled)
        self.num_features = features_scaled.shape[1]
        self.state_dim = window_size * self.num_features

        self.t = None
        self.steps_taken = None
        self.start_index = None

    def reset(self):
        max_start = self.num_steps - self.episode_length - 2

        if max_start <= self.window_size:
            raise ValueError("Dataset is too small for the chosen window_size and episode_length.")

        self.start_index = np.random.randint(self.window_size, max_start)
        self.t = self.start_index
        self.steps_taken = 0

        return self._get_state()

    def _get_state(self):
        past_window = self.features_scaled[self.t - self.window_size:self.t]
        return past_window.flatten()

    def step(self, action):
        next_pm25 = self.pm25_values[self.t + 1]

        if action == 1:
            reward = -0.1
        elif action == 0 and next_pm25 > 150:
            reward = -10.0
        else:
            reward = 0.0

        self.t += 1
        self.steps_taken += 1

        done = self.steps_taken >= self.episode_length

        next_state = self._get_state()

        return next_state, reward, done



# 5. Policy network


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.net(state)


# =========================
# 6. Compute discounted returns


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


# =========================
# 7. Train using REINFORCE


env = AirQualityEnv(
    features_scaled=features_scaled,
    pm25_values=pm25_values,
    window_size=24,
    episode_length=200
)

policy = PolicyNetwork(env.state_dim)

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

num_episodes = 500
gamma = 0.99

episode_returns = []

for episode in range(num_episodes):
    state = env.reset()

    log_probs = []
    rewards = []

    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        prob_on = policy(state_tensor)
        dist = Bernoulli(prob_on)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_int = int(action.item())

        next_state, reward, done = env.step(action_int)

        log_probs.append(log_prob.squeeze())
        rewards.append(reward)

        state = next_state

    total_return = sum(rewards)
    episode_returns.append(total_return)

    returns = compute_returns(rewards, gamma)

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 50 == 0:
        avg_return = np.mean(episode_returns[-50:])
        print(f"Episode {episode + 1}/{num_episodes}, Average Return: {avg_return:.2f}")


# =========================
# 8. Plot episodic return


plt.figure(figsize=(10, 5))
plt.plot(episode_returns)
plt.xlabel("Episode")
plt.ylabel("Total Episodic Return")
plt.title("REINFORCE on Air Quality Purifier Control")
plt.grid(True)
plt.savefig('return.jpeg')


# =========================
# 9. Smooth plot


window = 20
smooth_returns = pd.Series(episode_returns).rolling(window).mean()

plt.figure(figsize=(10, 5))
plt.plot(smooth_returns)
plt.xlabel("Episode")
plt.ylabel("Smoothed Total Return")
plt.title("Smoothed Episodic Return over 500 Episodes")
plt.grid(True)
plt.savefig('smooth.jpeg')


# =========================
# 10. Test trained policy


state = env.reset()
done = False

test_rewards = []
actions_taken = []

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob_on = policy(state_tensor)

    action = 1 if prob_on.item() >= 0.5 else 0

    next_state, reward, done = env.step(action)

    test_rewards.append(reward)
    actions_taken.append(action)

    state = next_state

print("\nTest total return:", sum(test_rewards))
print("Number of times purifier ON:", sum(actions_taken))
print("Number of times purifier OFF:", len(actions_taken) - sum(actions_taken))