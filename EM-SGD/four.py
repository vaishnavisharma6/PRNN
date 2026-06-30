import numpy as np
import pandas as pd
import time

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("dataset_4.csv")

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

n, d = X.shape
classes = np.unique(y)

# train test split
X_train = X[:8000]
y_train = y[:8000]

X_test = X[8000:10000]
y_test = y[8000:10000]


# =========================
# STANDARD SCALER
# =========================

class StandardScaler:

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# =========================
# LOGISTIC REGRESSION
# =========================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_logistic(X, y, lr=0.01, max_iter=10000, tol=1e-6):

    n, d = X.shape

    w = np.zeros(d)
    b = 0

    prev_loss = np.inf

    for i in range(max_iter):

        z = X @ w + b
        p = sigmoid(z)

        loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))

        if abs(prev_loss - loss) < tol:
            return w, b, i

        prev_loss = loss

        grad_w = X.T @ (p - y) / n
        grad_b = np.mean(p - y)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, max_iter


def train_ovr(X, y):

    weights = []
    biases = []
    iterations = []

    for c in classes:

        y_binary = (y == c).astype(int)

        w, b, it = train_logistic(X, y_binary)

        weights.append(w)
        biases.append(b)
        iterations.append(it)

    return np.array(weights), np.array(biases), iterations


def predict_ovr(X, weights, biases):

    scores = []

    for w, b in zip(weights, biases):
        scores.append(sigmoid(X @ w + b))

    scores = np.array(scores)

    preds = np.argmax(scores, axis=0)

    return classes[preds]


# =========================
# KNN LOOP VERSION
# =========================

def knn_loops(X_train, y_train, X_test, k=5):

    preds = []

    for x in X_test:

        dists = []

        for xi in X_train:
            d = np.sqrt(np.sum((x - xi)**2))
            dists.append(d)

        dists = np.array(dists)

        idx = np.argsort(dists)[:k]

        labels = y_train[idx]

        vals, counts = np.unique(labels, return_counts=True)

        preds.append(vals[np.argmax(counts)])

    return np.array(preds)


# =========================
# VECTORIZED KNN
# =========================

def knn_vectorized(X_train, y_train, X_test, k=5):

    X2 = np.sum(X_test**2, axis=1, keepdims=True)
    Y2 = np.sum(X_train**2, axis=1)

    cross = X_test @ X_train.T

    dists = np.sqrt(X2 + Y2 - 2*cross)

    preds = []

    for row in dists:

        idx = np.argsort(row)[:k]

        labels = y_train[idx]

        vals, counts = np.unique(labels, return_counts=True)

        preds.append(vals[np.argmax(counts)])

    return np.array(preds)


# =========================
# ACCURACY
# =========================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# =========================
# GAUSSIAN NAIVE BAYES
# =========================

def train_gnb(X, y):

    classes = np.unique(y)

    priors = {}
    means = {}
    vars_ = {}

    for c in classes:

        Xc = X[y == c]

        priors[c] = len(Xc) / len(X)

        means[c] = np.mean(Xc, axis=0)

        vars_[c] = np.var(Xc, axis=0) + 1e-9

    return priors, means, vars_


def predict_gnb_log(X, priors, means, vars_):

    preds = []

    for x in X:

        scores = []

        for c in priors:

            mean = means[c]
            var = vars_[c]

            log_likelihood = -0.5*np.sum(
                np.log(2*np.pi*var) + ((x-mean)**2)/var
            )

            score = np.log(priors[c]) + log_likelihood

            scores.append(score)

        preds.append(list(priors.keys())[np.argmax(scores)])

    return np.array(preds)


# =========================
# UNDERFLOW TEST
# =========================

def find_underflow(X, means, vars_, max_d=64):

    x = X[0]

    for D in range(1, max_d):

        prob = 1.0

        for j in range(D):

            mu = means[list(means.keys())[0]][j]
            var = vars_[list(vars_.keys())[0]][j]

            p = (1/np.sqrt(2*np.pi*var)) * np.exp(-(x[j]-mu)**2/(2*var))

            prob *= p

        if prob == 0.0:
            print("Underflow occurs at dimension:", D)
            return

    print("No underflow up to dimension", max_d)


# =========================================================
# 4.12 MULTICLASS LOGISTIC REGRESSION
# =========================================================

print("\n---- Logistic Regression (Raw Data) ----")

w_raw, b_raw, it_raw = train_ovr(X_train, y_train)

print("Iterations (raw):", it_raw)

print("\n---- Logistic Regression (Scaled Data) ----")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

w_scaled, b_scaled, it_scaled = train_ovr(X_train_scaled, y_train)

print("Iterations (scaled):", it_scaled)


# =========================================================
# 4.13 VECTORISATION BENCHMARK
# =========================================================

print("\n---- KNN Benchmark ----")

start = time.time()
knn_loops(X_train, y_train, X_test)
loop_time = time.time() - start

start = time.time()
knn_vectorized(X_train, y_train, X_test)
vec_time = time.time() - start

print("Loop time:", loop_time)
print("Vectorized time:", vec_time)
print("Speedup factor:", loop_time / vec_time)


# =========================================================
# 4.14 CURSE OF SCALE
# =========================================================

print("\n---- KNN Accuracy Comparison ----")

pred_raw = knn_vectorized(X_train, y_train, X_test)
acc_raw = accuracy(y_test, pred_raw)

pred_scaled = knn_vectorized(X_train_scaled, y_train, X_test_scaled)
acc_scaled = accuracy(y_test, pred_scaled)

print("Accuracy raw:", acc_raw)
print("Accuracy scaled:", acc_scaled)
print("Difference:", acc_scaled - acc_raw)


# =========================================================
# 4.15 NAIVE BAYES UNDERFLOW
# =========================================================

print("\n---- Gaussian Naive Bayes ----")

priors, means, vars_ = train_gnb(X_train, y_train)

pred_nb = predict_gnb_log(X_test, priors, means, vars_)

acc_nb = accuracy(y_test, pred_nb)

print("Naive Bayes accuracy:", acc_nb)

print("\n---- Underflow Test ----")

find_underflow(X_train, means, vars_)