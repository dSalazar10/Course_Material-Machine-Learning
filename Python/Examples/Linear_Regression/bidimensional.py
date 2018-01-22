import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

nb_samples = 100

def f(x):
    # slope or β[1]
    coeff = 25
    # y-intercept or β[0]
    bias = 3
    # True regression line value; y = β[1] * X + β[0]
    y = coeff * x  + bias
    # ε is a random variable assumed to be N(0, σ^2)
    error = y * random.uniform(-0.01, 0.01) 
    # f(x) = βX + β[0] + ε
    return y + error

def get_data():
    values = []
    # generate 300 random point that we will use to train a model
    for i in range(0, 300):
        # independent variable
        x = random.uniform(1, 1000)
        # dependent variable
        y = f(x)
        values.append((x, y))
    # split the values into two series instead of a list of tuples
    return values

def loss(v):
   e = 0.0
   for i in range(nb_samples):
      e += np.square(v[0] + v[1]*X[i] - Y[i])
   return 0.5 * e

def gradient(v):
   g = np.zeros(shape=2)
   for i in range(nb_samples):
     g[0] += (v[0] + v[1]*X[i] - Y[i])
     g[1] += ((v[0] + v[1]*X[i] - Y[i]) * X[i])
   return g

X, Y = zip(*get_data())
print(minimize(fun=loss, x0=[0.0, 0.0], jac=gradient, method='L-BFGS-B'))
# Display the training data points
plt.scatter(X, Y, color='blue')
plt.title('r = 1 Perfect Possitive Correlation', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
