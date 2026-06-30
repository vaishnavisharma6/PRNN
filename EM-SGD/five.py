import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load Dataset 1
# -------------------------------------------------

data = pd.read_csv("dataset_1.csv")

x = data.iloc[:,0].to_numpy()
y = data.iloc[:,1].to_numpy()

x = x.reshape(-1,1)

# -------------------------------------------------
# Polynomial Feature Generator
# -------------------------------------------------

def polynomial_features(x, degree):

    n = x.shape[0]

    X = np.ones((n, degree+1))

    for d in range(1, degree+1):
        X[:,d] = x[:,0]**d

    return X


# -------------------------------------------------
# Linear Regression (closed form)
# -------------------------------------------------

def linear_regression(X, y):

    w = np.linalg.pinv(X.T @ X) @ X.T @ y

    return w


# -------------------------------------------------
# Prediction
# -------------------------------------------------

def predict(X, w):

    return X @ w


# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

np.random.seed(0)

n = len(x)

perm = np.random.permutation(n)

train_size = int(0.7*n)

train_idx = perm[:train_size]
test_idx = perm[train_size:]


x_train = x[train_idx]
y_train = y[train_idx]

train_mean = np.mean(x_train)
train_std = np.std(x_train)

x_train = (x_train - train_mean) / train_std #normalization


x_test = x[test_idx]
y_test = y[test_idx]

mean_test = np.mean(x_test)
std_test = np.std(x_test)

x_test = (x_test - mean_test) / std_test  #normalization


# -------------------------------------------------
# Bias Variance via Bootstrapping
# -------------------------------------------------

B = 100

degrees = [1,15]

results = {}

for d in degrees:

    X_train = polynomial_features(x_train,d)
    X_test = polynomial_features(x_test,d)

    preds = []

    for b in range(B):

        idx = np.random.choice(len(x_train),
                               len(x_train),
                               replace=True)

        X_boot = X_train[idx]
        y_boot = y_train[idx]

        w = linear_regression(X_boot,y_boot)

        pred = predict(X_test,w)

        preds.append(pred)

    preds = np.array(preds)

    mean_pred = np.mean(preds,axis=0)

    bias = np.mean((mean_pred - y_test)**2)

    variance = np.mean(np.var(preds,axis=0))

    results[d] = (bias,variance)


print("\nBias Variance Results")
print("Degree 1 -> Bias, Variance:",results[1])
print("Degree 15 -> Bias, Variance:",results[15])


# -------------------------------------------------
# Frequentist Variance of Slope (bootstrap)
# -------------------------------------------------

slopes = []

for b in range(B):

    idx = np.random.choice(len(x_train),
                           len(x_train),
                           replace=True)

    X_boot = polynomial_features(x_train,1)[idx]
    y_boot = y_train[idx]

    w = linear_regression(X_boot,y_boot)

    slopes.append(w[1])


slopes = np.array(slopes)

freq_mean = np.mean(slopes)
freq_var = np.var(slopes)

print("\nFrequentist slope mean:",freq_mean)
print("Frequentist slope variance:",freq_var)


# -------------------------------------------------
# Bayesian MAP Estimator
# -------------------------------------------------

X_train = polynomial_features(x_train,1)

sigma2 = 1
sigma_prior2 = 10

I = np.eye(X_train.shape[1])

Sigma_post = np.linalg.inv(
    (1/sigma2)*(X_train.T @ X_train) +
    (1/sigma_prior2)*I
)

mu_post = Sigma_post @ ((1/sigma2)*(X_train.T @ y_train))

posterior_variance_slope = Sigma_post[1,1]

print("\nMAP estimate:",mu_post)
print("Posterior variance of slope:",posterior_variance_slope)


# -------------------------------------------------
# Plot Frequentist vs Bayesian
# -------------------------------------------------

plt.hist(slopes,
         bins=20,
         density=True,
         alpha=0.6,
         label="Frequentist")

x_vals = np.linspace(freq_mean-3*np.sqrt(freq_var),
                     freq_mean+3*np.sqrt(freq_var),
                     200)

bayes_std = np.sqrt(posterior_variance_slope)

bayes_pdf = (1/(bayes_std*np.sqrt(2*np.pi))) * np.exp(
    -(x_vals-mu_post[1])**2/(2*bayes_std**2)
)

plt.plot(x_vals,
         bayes_pdf,
         linewidth=2,
         label="Bayesian Posterior")

plt.xlabel("Slope Value")
plt.ylabel("Density")
plt.title("Frequentist vs Bayesian Uncertainty")

plt.legend()

plt.savefig('histo.jpeg')