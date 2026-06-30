import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("/Users/vaishnavisharma/prnn-3/delhi_aqi.csv")

if "date" in df.columns:
    df = df.drop(columns=["date"])

print("Columns in dataset:")
print(df.columns)


# -----------------------------
# Target: continuous PM2.5
# -----------------------------
target_col = "pm2_5"

y = df[target_col].values.astype(float)

X = df.drop(columns=[target_col])

# Convert categorical columns if present
X = pd.get_dummies(X)

# Fill missing values
X = X.fillna(X.mean())

X = X.values.astype(float)


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Gradient Boosting
# -----------------------------
n_estimators = 50
learning_rate = 0.1
max_depth = 8

# Initial prediction = mean of target
initial_prediction = np.mean(y_train)

train_pred = np.ones(len(y_train)) * initial_prediction
test_pred = np.ones(len(y_test)) * initial_prediction

models = []
residual_variances = {}
stages_to_store = [1, 5, 10, 50]

for m in range(1, n_estimators + 1):

    print(f"iteration {m}/{n_estimators}")

    #  Compute residuals
    residuals = y_train - train_pred

    #  Train weak learner on residuals
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=m
    )

    tree.fit(X_train, residuals)

    #  Predict residual correction
    train_update = tree.predict(X_train)
    test_update = tree.predict(X_test)

    #  Update predictions
    train_pred += learning_rate * train_update
    test_pred += learning_rate * test_update

    #  Compute new residuals
    new_residuals = y_train - train_pred

    #  Store residual variance
    if m in stages_to_store:
        residual_variances[m] = np.var(new_residuals)
        print(f"Residual variance at stage {m}: {residual_variances[m]}")


#evaluate
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print("\nFinal Results")
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)



# Plot residual variance

plt.figure(figsize=(8, 5))

plt.plot(
    list(residual_variances.keys()),
    list(residual_variances.values()),
    marker="o"
)

plt.xlabel("Boosting Stage")
plt.ylabel("Residual Variance")
plt.title("Residual Variance after Gradient Boosting Stages")
plt.grid(True)
plt.savefig('res_var.jpeg')