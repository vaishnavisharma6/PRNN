import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

data = pd.read_csv("dataset_2.csv")

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy().reshape(-1,1)

n, d = X.shape

print("Samples:", n)
print("Features:", d)

# -------------------------------------------------------
# Normalize features (important for Lasso convergence)
# -------------------------------------------------------

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# -------------------------------------------------------
# 2.3 Attempt OLS
# -------------------------------------------------------

print("\nAttempting OLS solution...")

try:
    XtX = X.T @ X

    rank = np.linalg.matrix_rank(XtX)
    if rank < XtX.shape[0]:
        raise np.linalg.LinAlgError("Singular matrix: X^T X is rank deficient")

    w_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    print('w_ols:', w_ols)
except np.linalg.LinAlgError as e:
    print("Exception encountered:", e)

# -------------------------------------------------------
# Ridge Regression
# -------------------------------------------------------

def ridge_solution(X, y, lam):

    d = X.shape[1]
    I = np.eye(d)

    w = np.linalg.inv(X.T @ X + lam*I) @ X.T @ y

    return w


# -------------------------------------------------------
# Condition number vs lambda
# -------------------------------------------------------
w = ridge_solution(X, y, lam = 0.1)
print('solution:', w)
print("\nComputing condition numbers...")

lambdas = np.logspace(-5,2,25)
condition_numbers = []

for lam in lambdas:

    A = X.T @ X + lam*np.eye(d)

    s = np.linalg.svd(A, compute_uv=False)

    cond = s.max() / s.min()

    condition_numbers.append(cond)

plt.figure()
plt.plot(lambdas, condition_numbers)
plt.xlabel("lambda")
plt.ylabel("Condition Number")
plt.title("Condition Number of (X^T X + λI)")
plt.savefig('lambda.jpeg')


# -------------------------------------------------------
# 2.4 Lasso using Coordinate Descent
# -------------------------------------------------------

def soft_threshold(z, lam):

    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    else:
        return 0


def lasso_coordinate_descent(X, y, lam, max_iter=40):

    n, d = X.shape
    w = np.zeros(d)

    y = y.flatten()

    for _ in range(max_iter):

        for j in range(d):

            residual = y - X @ w + X[:,j]*w[j]

            rho = X[:,j] @ residual

            w[j] = soft_threshold(rho/n, lam)

    return w


# -------------------------------------------------------
# Regularization path
# -------------------------------------------------------

print("\nRunning Lasso regularization path...")

lasso_lambdas = np.logspace(-5,2,20)

weights = []

lambda_50 = None

for lam in lasso_lambdas:

    w = lasso_coordinate_descent(X, y, lam)

    weights.append(w)

    zero_fraction = np.sum(np.abs(w) < 1e-6) / len(w)

    if lambda_50 is None and zero_fraction >= 0.5:
        lambda_50 = lam

weights = np.array(weights)

# -------------------------------------------------------
# Plot regularization path
# -------------------------------------------------------

plt.figure()

for j in range(weights.shape[1]):
    plt.plot(np.log(lasso_lambdas), weights[:,j])

plt.xlabel("log(lambda)")
plt.ylabel("Weight values")
plt.title("Lasso Regularization Path")

plt.savefig('path.jpeg')


# -------------------------------------------------------
# Report lambda where 50% weights become zero
# -------------------------------------------------------

print("\nLambda where approximately 50% weights become zero:", lambda_50)