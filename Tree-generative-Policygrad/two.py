import numpy as np
import pandas as pd


# =====================================================
# GINI FUNCTIONS
# =====================================================
def gini_impurity(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y)
    p0 = 1 - p1
    return 1 - (p0**2 + p1**2)


def gini_gain(parent, left, right):
    n = len(parent)
    return gini_impurity(parent) - (
        (len(left)/n)*gini_impurity(left) +
        (len(right)/n)*gini_impurity(right)
    )


def find_best_threshold(X_column, y):
    if len(X_column) <= 1:
        return None, -np.inf

    sorted_idx = np.argsort(X_column)
    X_sorted = X_column[sorted_idx]
    y_sorted = y[sorted_idx]

    best_gain = -np.inf
    best_threshold = None

    for i in range(1, len(X_sorted)):

        if X_sorted[i] == X_sorted[i-1]:
            continue

        threshold = (X_sorted[i] + X_sorted[i-1]) / 2

        left = y_sorted[X_sorted <= threshold]
        right = y_sorted[X_sorted > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        gain = gini_gain(y_sorted, left, right)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


# =====================================================
# DECISION TREE (SAFE VERSION)
# =====================================================
class DecisionTree:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def majority_class(self, y):
        if len(y) == 0:
            return 0
        return np.bincount(y).argmax()

    def best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for f in range(X.shape[1]):
            threshold, gain = find_best_threshold(X[:, f], y)

            if threshold is not None and gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold

        return best_feature, best_threshold

    def build(self, X, y, depth):

       
        if len(y) == 0:
            return 0

        # Pure node
        if len(set(y)) == 1:
            return y[0]

        # Max depth
        if depth >= self.max_depth:
            return self.majority_class(y)

        feature, threshold = self.best_split(X, y)

        # No valid split
        if feature is None or threshold is None:
            return self.majority_class(y)

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # Bad split
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self.majority_class(y)

        left_subtree = self.build(X[left_mask], y[left_mask], depth+1)
        right_subtree = self.build(X[right_mask], y[right_mask], depth+1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def fit(self, X, y):
        self.tree = self.build(X, y, 0)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


# =====================================================
# RANDOM FOREST
# =====================================================
class RandomForest:

    def __init__(self, n_trees=10, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        n = len(X)
        idx = np.random.choice(n, n, replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):

            X_sample, y_sample = self.bootstrap_sample(X, y)

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            final_preds.append(np.bincount(votes).argmax())

        return np.array(final_preds)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    df = pd.read_csv("/Users/vaishnavisharma/prnn-3/delhi_aqi.csv")

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Remove missing values
    df = df.dropna()

    # Labels
    y = (df["pm2_5"] > 200).astype(int).values

    # All features except PM2.5
    X = df.drop(columns=["date", "pm2_5"]).values

    # Train
    rf = RandomForest(n_trees=5, max_depth=3)
    rf.fit(X, y)

    # Predict
    preds = rf.predict(X)

    # Accuracy
    acc = np.mean(preds == y)

    print("Accuracy:", acc)