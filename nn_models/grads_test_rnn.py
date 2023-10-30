
import numpy as np

from networks import recurrentNeuralNetwork


# hyperparams
m = 100
k1 = 7
k2 = 1
T = 20
lambd = 0.0
sigma = 0
eps = 0.0001

seed = 1
optimizer = None
N = 20

# model
model = recurrentNeuralNetwork(
    m,
    k1,
    k2,
    seed,
)

# get fake data, (N x T x D)
X_t = np.random.normal(size=(N, T, k1))
X_c = np.random.normal(size=(N, k2))
Y = np.random.normal(size=(N, k2))

# get loss for batch
loss = model.compute_loss(X_t, X_c, Y, lambd)

# compute grads analytically
grads_analytical = model.compute_grads(X_t, X_c, Y, sigma, lambd)

# compute grads numerically
grads_numerical = model.compute_grads_numerical(X_t, X_c, Y, sigma, lambd, eps)