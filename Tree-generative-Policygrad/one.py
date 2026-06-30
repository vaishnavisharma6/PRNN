import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv('/Users/vaishnavisharma/prnn-3/delhi_aqi.csv')
Y = np.where(df['pm2_5'] > 200, 1, 0)

def gini_impurity(Y):
    if len(Y) == 0:
        return 0
    
    p1 = np.mean(Y)
    p0 = 1 - p1

    return(1- (p0**2 + p1**2))


def gini_gain(parent, left, right):
    n = len(parent)

    g_parent = gini_impurity(parent)
    g_left = gini_impurity(left)
    g_right = gini_impurity(right)

    weighted = (len(left)/n) * g_left + (len(right)/n)* g_right

    reduction = g_parent - weighted

    return reduction

def best(X, Y):
    sorted_idx = np.argsort(X)
    X_sorted = X[sorted_idx]
    Y_sorted = Y[sorted_idx]


    best_gain = -np.inf
    best_threshold = None

    for i in range(1, len(X_sorted)):
        if X_sorted[i] == X_sorted[i-1]:
            continue

        threshold = (X_sorted[i]+X_sorted[i-1])/2
        left = X_sorted <= threshold
        right = X_sorted > threshold

        left_Y = Y_sorted[left]
        right_Y = Y_sorted[right]

        gain = gini_gain(Y_sorted, left_Y, right_Y)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return(best_threshold, best_gain)        


X = df['co'].values # iteration over co values
threshold, gain = best(X, Y)
print('best threshold:', threshold)
print('best gain:', gain)
