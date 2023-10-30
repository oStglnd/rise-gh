
import numpy as np

from nn import neuralNetwork


# hyperparams
K = 1
m = [16, 32, 32, 4]
k1 = 7
k2 = 1
lambd = 0.1
sigma = 0.0
eps = 0.0001
eta = 0.001

seed = 1
optimizer = None
N = 20

# model
model = neuralNetwork(
    k1=k1,
    k2=k2,
    m=m,
    initialization='He',
    optimizer=optimizer,
    seed=seed
)


# get fake data, (N x T x D)
X_t = np.random.normal(size=(N, k1))
X_c = np.random.normal(size=(N, k2))
Y = np.random.normal(size=(N, K))

# get loss for batch
loss = model.compute_loss(X_t, X_c, Y, lambd)

# compute grads analytically
grads_analytical = model.compute_grads(X_t, X_c, Y, sigma, lambd)

# # compute grads numerically
grads_numerical = model.compute_grads_numerical(X_t, X_c, Y, sigma, lambd, eps)