
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load Dataset

np.random.seed(0)
data = pd.read_csv("dataset_3.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values



# Utility


def sigmoid(z):
    z = np.clip(z,-50,50)
    return 1/(1+np.exp(-z))


############################################
# 3.5 Logistic Regression


mask = (y==1)|(y==2)
X_lr = X[mask]
y_lr = y[mask]
y_lr = (y_lr==2).astype(int)

def logistic_regression(X,y,lr=0.01,lam=0.01,iters=500):

    n,d = X.shape
    w = np.zeros(d)

    losses=[]

    for i in range(iters):

        pred = sigmoid(X@w)
        pred = np.clip(pred,1e-8,1-1e-8)

        loss = -(1/n)*(y*np.log(pred)+(1-y)*np.log(1-pred)).sum()
        loss += (lam/2)*np.sum(w*w)

        grad = (1/n)*(X.T@(pred-y)) + lam*w

        w -= lr*grad

        losses.append(loss)

    return w,losses


w,losses = logistic_regression(X_lr,y_lr)
learning_rates = [0.01, 0.5, 5.0] # 5.0 is likely to violate the condition
plt.figure(figsize=(10, 6))

for rate in learning_rates:
    _, experiment_losses = logistic_regression(X_lr, y_lr, lr=rate, lam=0.1, iters=100)
    plt.plot(experiment_losses, label=f"lr = {rate}")

plt.title("Effect of Learning Rate on Convergence (Lipschitz Violation)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.yscale("log") # Using log scale to see the explosion of loss clearly
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("logistic_lipschitz_violation.png", dpi=300)
plt.savefig('lr.jpeg')
plt.figure()
plt.plot(losses)
plt.title("Logistic Regression Learning Curve")
plt.savefig("logistic_curve.png",dpi=300)



############################################
# Gaussian PDF


def gaussian_pdf(X,mu,cov):

    d = X.shape[1]

    cov = cov + 1e-6*np.eye(d)

    inv = np.linalg.inv(cov)

    det = np.linalg.det(cov)

    diff = X-mu

    exponent = np.sum((diff@inv)*diff,axis=1)

    norm = 1/np.sqrt((2*np.pi)**d * det)

    return norm*np.exp(-0.5*exponent)


############################################
# Log Likelihood


def compute_loglik(X,mus,covs,pis):

    n = X.shape[0]
    K = len(pis)

    probs = np.zeros((n,K))

    for k in range(K):
        probs[:,k] = pis[k]*gaussian_pdf(X,mus[k],covs[k])

    return np.sum(np.log(probs.sum(axis=1)+1e-12))


############################################
# 3.6 Initial Log Likelihood and One EM Iter


K = 3
n,d = X.shape

mus = X[np.random.choice(n,K,replace=False)]
covs = [np.cov(X.T)+1e-6*np.eye(d) for _ in range(K)]
pis = np.ones(K)/K

initial_ll = compute_loglik(X,mus,covs,pis)

print("Initial Log Likelihood:",initial_ll)


# ----- E Step -----

probs = np.zeros((n,K))

for k in range(K):
    probs[:,k] = pis[k]*gaussian_pdf(X,mus[k],covs[k])

probs_sum = probs.sum(axis=1,keepdims=True)+1e-12

gamma = probs/probs_sum


# ----- M Step -----

Nk = gamma.sum(axis=0)

for k in range(K):

    mus[k] = (gamma[:,k][:,None]*X).sum(axis=0)/Nk[k]

    diff = X-mus[k]

    covs[k] = ((gamma[:,k][:,None]*diff).T@diff)/Nk[k] + 1e-6*np.eye(d)

    pis[k] = Nk[k]/n


after_one_ll = compute_loglik(X,mus,covs,pis)

print("Log Likelihood after one EM iteration:",after_one_ll)


############################################
# 3.7 EM Convergence


def gmm_em(X,K,iters=30):

    n,d = X.shape

    mus = X[np.random.choice(n,K,replace=False)]
    covs = [np.cov(X.T)+1e-6*np.eye(d) for _ in range(K)]
    pis = np.ones(K)/K

    loglik=[]

    for t in range(iters):

        probs = np.zeros((n,K))

        for k in range(K):
            probs[:,k] = pis[k]*gaussian_pdf(X,mus[k],covs[k])

        probs_sum = probs.sum(axis=1,keepdims=True)+1e-12

        gamma = probs/probs_sum

        Nk = gamma.sum(axis=0)

        for k in range(K):

            mus[k] = (gamma[:,k][:,None]*X).sum(axis=0)/Nk[k]

            diff = X-mus[k]

            covs[k] = ((gamma[:,k][:,None]*diff).T@diff)/Nk[k] + 1e-6*np.eye(d)

            pis[k] = Nk[k]/n

        ll = np.sum(np.log(probs_sum))

        loglik.append(ll)

    return mus,covs,pis,loglik


plt.figure()

for K in [2,4,6]:

    mus,covs,pis,ll = gmm_em(X,K)

    plt.plot(ll,label=f"K={K}")

plt.legend()
plt.title("EM Log Likelihood Convergence")
plt.savefig("em_convergence.png",dpi=300)



############################################
# 3.8 Decision Boundaries


X2 = X[:,:2]

mus,covs,pis,_ = gmm_em(X2,3)

xx,yy = np.meshgrid(
np.linspace(X2[:,0].min(),X2[:,0].max(),120),
np.linspace(X2[:,1].min(),X2[:,1].max(),120)
)

grid = np.c_[xx.ravel(),yy.ravel()]

def gmm_predict(X,mus,covs,pis):

    probs=[]

    for k in range(len(pis)):
        probs.append(pis[k]*gaussian_pdf(X,mus[k],covs[k]))

    probs = np.column_stack(probs)

    return np.argmax(probs,axis=1)


preds = gmm_predict(grid,mus,covs,pis)
preds = preds.reshape(xx.shape)

plt.figure()
plt.contourf(xx,yy,preds,alpha=0.3)
plt.scatter(X2[:,0],X2[:,1],s=5,c='black')
plt.title("GMM Decision Boundary")
plt.savefig("gmm_boundary.png",dpi=300)



############################################
# 3.9 Soft Margin SVM Dual


mask = (y==1)|(y==2)

X_svm = X[mask]
y_svm = y[mask]

y_svm = np.where(y_svm==1,-1,1)

n = len(y_svm)

K = X_svm@X_svm.T

C = 1.0
alpha = np.zeros(n)

lr=0.001

for it in range(500):

    grad = 1 - (alpha*y_svm)@(K*(y_svm[:,None]))

    alpha += lr*grad

    alpha = np.clip(alpha,0,C)


############################################
# Recover w and Check KKT


w = np.sum((alpha*y_svm)[:,None]*X_svm,axis=0)

margin = y_svm*(X_svm@w)

inside=[]
on=[]
outside=[]

for i in range(n):

    if margin[i] > 1:
        outside.append((i,alpha[i]))

    elif abs(margin[i]-1)<1e-3:
        on.append((i,alpha[i]))

    else:
        inside.append((i,alpha[i]))

print("Inside margin:",inside[:3])
print("On margin:",on[:3])
print("Outside margin:",outside[:3])


############################################
# 3.10 Mercer Condition


def rbf_kernel(X,sigma):

    sq = np.sum(X**2,axis=1)

    dist = sq[:,None] + sq[None,:] - 2*X@X.T

    K = np.exp(-dist/(2*sigma**2))

    return K


N = min(400,len(X))

X_sub = X[:N]

Kmat = rbf_kernel(X_sub,1)

eigvals = np.linalg.eigvalsh(Kmat)

print("Minimum eigenvalue:",np.min(eigvals))


############################################
# 3.11 Hyperparameter Topography


Cs=[0.1,1,10,100]
sigmas=[0.001,0.01,0.1,1,10]

heat=np.zeros((len(Cs),len(sigmas)))

for i,C in enumerate(Cs):
    for j,s in enumerate(sigmas):

        acc=np.random.rand()   # placeholder

        heat[i,j]=acc


plt.figure()
plt.imshow(heat)
plt.colorbar()

plt.xticks(range(len(sigmas)),sigmas)
plt.yticks(range(len(Cs)),Cs)

plt.xlabel("sigma")
plt.ylabel("C")
plt.title("Hyperparameter Topography")

plt.savefig("svm_heatmap.png",dpi=300)
