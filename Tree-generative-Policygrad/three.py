import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# Weighted Decision Stump
# -----------------------------
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float("inf")

        for feature in range(n_features):
            values = X[:, feature]
            thresholds = np.unique(values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)

                    if polarity == 1:
                        predictions[values <= threshold] = -1
                    else:
                        predictions[values > threshold] = -1

                    error = np.sum(sample_weights[predictions != y])

                    if error < min_error:
                        min_error = error
                        self.feature_index = feature
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        n_samples = X.shape[0]
        values = X[:, self.feature_index]

        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[values <= self.threshold] = -1
        else:
            predictions[values > self.threshold] = -1

        return predictions


# -----------------------------
# AdaBoost from Scratch
# -----------------------------
class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.weight_history = []
        self.misclassified_history = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        sample_weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            print(f"Boosting iteration {t + 1}/{self.n_estimators}")

            stump = DecisionStump()
            stump.fit(X, y, sample_weights)

            predictions = stump.predict(X)

            misclassified = predictions != y

            error = np.sum(sample_weights[misclassified])
            error = np.clip(error, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - error) / error)

            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.models.append(stump)
            self.alphas.append(alpha)

            self.weight_history.append(sample_weights.copy())
            self.misclassified_history.append(misclassified.copy())

    def predict(self, X):
        final_prediction = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            final_prediction += alpha * model.predict(X)

        return np.sign(final_prediction)


# -----------------------------
# Main Code
# -----------------------------

# Change filename if needed
df = pd.read_csv("/Users/vaishnavisharma/prnn-3/delhi_aqi.csv")

# Binary classification target
df["target"] = (df["pm2_5"] > 200).astype(int)

# Convert labels from {0, 1} to {-1, +1}
y = df["target"].values
y = np.where(y == 1, 1, -1)

# Features
X = df.drop(columns=["pm2_5", "date", "target"])

# Convert categorical columns, if any
X = pd.get_dummies(X)

# Fill missing values
X = X.fillna(X.mean())

X = X.values.astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train AdaBoost
model = AdaBoostScratch(n_estimators=50)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)


# -----------------------------
# Track 5 most misclassified samples
# -----------------------------
weight_history = np.array(model.weight_history)
misclassified_history = np.array(model.misclassified_history)

misclassification_counts = misclassified_history.sum(axis=0)

top5_indices = np.argsort(misclassification_counts)[-5:]

print("\nTop 5 most consistently misclassified training samples:")
print(top5_indices)

print("\nNumber of times misclassified:")
print(misclassification_counts[top5_indices])


# -----------------------------
# Plot weight growth
# -----------------------------
plt.figure(figsize=(9, 6))

for idx in top5_indices:
    plt.plot(weight_history[:, idx], label=f"Sample {idx}")

plt.xlabel("Boosting Iteration")
plt.ylabel("Sample Weight")
plt.title("Weight Growth of 5 Most Consistently Misclassified Samples")
plt.legend()
plt.grid(True)
plt.show()