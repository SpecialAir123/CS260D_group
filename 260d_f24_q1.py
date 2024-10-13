# -*- coding: utf-8 -*-
"""260D_F24_Q1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13r6OCrpnk3FF1XHxnPIdf1-SUf0WzkEL
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import random

def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)

def get_data():

    set_seed()

    X = np.random.uniform(-1, 1, size=(N, D))
    w = np.random.uniform(-1, 1, size=D)
    y = np.matmul(X, w) + np.random.normal(size=N)
    return w, X, y

def gradient(w, x_i, y_i):
    return 2*(np.dot(w, x_i) - y_i)*x_i

def get_loss(w, X, y):
    return np.sum((np.matmul(X, w) - y)**2)

def sgd(w_0, X, y):

    set_seed()

    w = w_0
    k = 0
    losses = [get_loss(w, X, y)]
    for s in range(NUM_EPOCHS):
        for t in range(N):
            idx = np.random.choice(N)
            w = w - lr/np.sqrt(k + 1)*gradient(w, X[idx], y[idx])
            #w = w - lr*gradient(w, X[idx], y[idx])
            losses.append(get_loss(w, X, y))
            k += 1
    return losses, w

def sgd_shuffle(w_0, X, y):

    set_seed()

    w = w_0
    losses = [get_loss(w, X, y)]
    k = 0
    for s in range(NUM_EPOCHS):
        ndx = list(range(N))
        random.shuffle(ndx)
        # ... fill here; don't introduce any randomness in your code
        for idx in ndx:
            # Update the weights using the gradient at X[idx], y[idx]
            w = w - lr / np.sqrt(k + 1) * gradient(w, X[idx], y[idx])
            losses.append(get_loss(w, X, y))
            k += 1
    return losses, w

def sgd_momentum(w_0, X, y, beta=0.5):

    set_seed()

    w = w_0
    delta_w = np.zeros_like(w)  # Initialize delta_w to zero
    k = 0
    losses = [get_loss(w, X, y)]

    for s in range(NUM_EPOCHS):
        ndx = list(range(N))
        random.shuffle(ndx)
        # ... fill here; don't introduce any randomness in your code
        for idx in ndx:
            # Compute the gradient at the current data point
            grad = gradient(w, X[idx], y[idx])

            # Update the momentum term
            delta_w = beta * delta_w + lr / np.sqrt(k + 1) * grad

            # Update the weights using the momentum term
            w = w - delta_w

            losses.append(get_loss(w, X, y))
            k += 1
    return losses, w

def plot(loss):
    for name, loss_val in loss.items():
        plt.plot(np.arange(len(loss_val)), loss_val, label=name)
    plt.legend()
    plt.savefig('loss.png')

def main():
    w_0, X, y = get_data()
    loss = {}
    loss['SGD'], w_sgd = sgd(w_0, X, y)
    loss['SGD-shuffle'], w_sgd_shuffle = sgd_shuffle(w_0, X, y)
    loss['SGD-momentum'], w_sgd_momentum = sgd_momentum(w_0, X, y)
    plot(loss)
    plt.show()

D = 10
N = 300
lr = 0.001
NUM_EPOCHS = 5

main()