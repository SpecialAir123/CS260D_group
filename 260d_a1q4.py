# -*- coding: utf-8 -*-
"""260D A1Q4

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Te-GR2PdYMQYAS2llQDqLQ6zHPHp8GIG
"""

import numpy as np
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
import time
import random

random.seed(42)
np.random.seed(24)

# Let's create the data
data_kwargs = {"n_samples":1_000_00, "n_features":12, "noise":3, "coef":True, "random_state":101}
X, y, coef = make_regression(**data_kwargs)

# Setting up PySpark
import pyspark
sc = pyspark.SparkContext().getOrCreate()

"""# "Mini-batch Gradient Descent with PySpark"

"""

# Fit a linear regression with sklean to compare the solution later
X, y, coef = make_regression(**data_kwargs)

# Parallelize the data to run distributed
data = np.hstack([X, y.reshape(-1,1)]).tolist()
rdd = sc.parallelize(data).cache()
print(f"Our RDD has {rdd.getNumPartitions()} partitions.")

"""Calculate the gradient of MSE loss for every example below."""

# define the gradient of MSE for every example
def per_example_mse_gradient(example, w):
    x = example[:-1]
    y = example[-1]
    # ... calculate the gradient here
    x = np.array(x)
    w = np.array(w)
    error = np.dot(x, w) - y  # Compute the prediction error
    weights_gradient = error * x  # Gradient: (x^T w - y) * x
    return weights_gradient

# calculate cumulative sum of gradients
def cum_sum_gradients(row, next_row):
    return [gradient+next_gradient for gradient, next_gradient in zip(row, next_row)]

"""Write the map and reduce function below, using the per_example_mse_gradient and cum_sum_gradients functions"""

def distributed_minibatch_gradient_descent(rdd, learning_rate=0.1, n_iters=100, mini_frac=0.1):
    w = np.zeros(len(rdd.first()) - 1).tolist()  # -1 because the last value is y
    print(rdd.count())
    for i in range(n_iters):
        mini_batch = rdd.sample(False,mini_frac)
        m = mini_batch.count()

        # fill out the map and reduce functions using the per_example and cum_sum functions defined above
        rdd_gradient = mini_batch.map(lambda example: per_example_mse_gradient(example, w)) \
            .reduce(cum_sum_gradients)

        # scaling with m and learning rate
        w_gradient = [learning_rate * (w / m) for w in rdd_gradient]

        # updating weights
        w = [w_j - w_grad_j for w_j, w_grad_j in zip(w, w_gradient)]

        print([i, m])

    return w

rdd = sc.parallelize(data).cache()
print(f"Our RDD has {rdd.getNumPartitions()} partitions.")

epoch=10
m_frac=0.4
n_iter = int(1.0/m_frac*epoch)
print(n_iter)
w = distributed_minibatch_gradient_descent(rdd, n_iters=n_iter, mini_frac=m_frac)

# Let's compare our solutoin with sklearn
w = [round(w_j, 2) for w_j in w]
coef = [round(c, 2) for c in coef]
print(f"""Here is a side-by-side comparison of our coefficients with Scikit-Learn's:
distributed mini-batch gradient descent -> {w}
Scikit-Learn's coefficients  -> {coef}""")

"""Write a distributed function for gradient descent (without mini-batching) below"""

# write a distributed function for gradient descent (without mini-batching)
def distributed_gradient_descent(rdd, learning_rate=0.1, epoch=10):
    w = np.zeros(len(rdd.first())-1).tolist() # -1 because the last value is y
    m = rdd.count()
    for i in range(epoch):
        rdd_gradient = rdd.map(lambda example: per_example_mse_gradient(example, w)) \
            .reduce(cum_sum_gradients)

        # scaling with m and learning rate
        w_gradient = [learning_rate*(w/m) for w in rdd_gradient]

        # updating weights
        w = [w_j - w_grad_j for w_j, w_grad_j in zip(w, w_gradient)]

    return w

# Train the model with distributed GD for 10 epochs
w_d = distributed_gradient_descent(rdd, epoch=10)
print(f"""Here is a side-by-side comparison of our coefficients with Scikit-Learn's:
distributed gradient descent -> {w_d}
Scikit-Learn's coefficients  -> {coef}""")

"""2. Compare your solution for distributed GD and distributed mini-batch DG with
that of Sklearn. What do you observe and why?
(Select all that are correct):
A. Distributed mini-batch GD is converging faster (less time)  
B. Distributed GD is converging faster  (less time)  
C. Distributed mini-batch GD has a closer solution to that of sklearn  
D. Distributed GD has a closer solution to that of sklearn  
E. Distributed GD and distributed mini-batch GD behave the same.  


3. Increase the number of iterations for distributed GD and compare the solutions. What do you observe?  
A. Distributed GD’s solution gets farther away from the sklearn solution  
B. Distributed GD’s solution gets closer to the sklearn solution  
C. More iterations do not change the output of Distributed GD.  

4. Change the mini-batch size to a larger batch size for distributed mini-batch GD and compare the solutions again. What do you observe?  
A. Distributed mini-batch GD now converges faster (less time)  
B. Distributed mini-batch GD now converges slower (less time)  
C. Change of batch size does not influence the behavior of Distributed mini-batch GD.  

5. Select the learning rate for distributed mini-batch GD with a batch size of 40\% (on the master node), that converges closest to the sklearn solution in 10 iterations.  
A. 0.1  
B. 0.01  
C. 0.001   
D. 0.5  
E. 1.0  
"""

# shutting down spark context
sc.stop()