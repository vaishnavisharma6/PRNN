import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# =========================
# Load Dataset
# =========================

data = pd.read_csv("dataset_1.csv")

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

n = len(x)

print("Number of samples:", n)

# =========================
# Phase 1.1 OLS
# =========================

# Design matrix
X = np.hstack([x, np.ones((n,1))])

# Closed form solution
w_closed = np.linalg.pinv(X.T @ X) @ X.T @ y

# Gradient Descent
def gradient_descent(X, y, lr=0.01, iterations=5000):

    n_samples, n_features = X.shape
    w = np.zeros((n_features,1))

    for _ in range(iterations):

        prediction = X @ w
        error = prediction - y
        gradient = (2/n_samples) * (X.T @ error)
        w = w - lr * gradient

    return w

w_gd = gradient_descent(X, y)

# L2 difference
l2_norm = np.linalg.norm(w_closed - w_gd)

print("\nClosed Form Weights:")
print(w_closed)

print("\nGradient Descent Weights:")
print(w_gd)

print("\nL2 Norm Difference:")
print(l2_norm)

# =========================
# Phase 1.2 SVM Dual
# =========================

# Convert to binary labels
median_val = np.median(y)
y_binary = np.where(y >= median_val, 1, -1).astype(float)

# Kernel matrix
K = x @ x.T

# Q matrix
Q = (y_binary @ y_binary.T) * K

# =========================
# Quadratic Programming Setup
# =========================

P = matrix(Q)
q = matrix(-np.ones(n))

# mu >= 0
G = matrix(-np.eye(n))
h = matrix(np.zeros(n))

# equality constraint sum(mu_i y_i) = 0
A = matrix(y_binary.reshape(1,-1))
b = matrix(np.zeros(1))

# =========================
# Solve QP
# =========================

solvers.options['show_progress'] = False

solution = solvers.qp(P, q, G, h, A, b)

mu = np.array(solution['x']).flatten()

# Support vectors
support_vectors = np.where(mu > 1e-2)[0]

print("\nSupport Vector Indices:")
print(support_vectors)
