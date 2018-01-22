import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt # plotting functions

# single array which contains both mean (0) and variance (1)
def negative_log_likelihood(v):
    l = 0.0
    f1 = 1.0 / np.sqrt(2.0 * np.pi * v[1]) 
    f2 = 2.0 * v[1]
    for x in X_data:
        l += np.log(f1 * np.exp(-np.square(x - v[0]) / f2))
    return -l

# 100 points drawn from a Gaussian distribution with zero mean and a standard
# deviation equal to 2.0
X_data = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=100)

# Display the 100 points
plt.plot(X_data)
plt.title('Gaussian Distribution', fontsize=16)
plt.xlabel('Points', fontsize=13)
plt.ylabel('Values', fontsize=13)



print(minimize(fun=negative_log_likelihood, x0=[0.0, 1.0]))
