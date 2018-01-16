# -*- coding: utf-8 -*-
"""
linear_regression.py

Daniel Salazar
Jan 12, 2018
CSCI 6820
Assignment 1 Prep

https://tutorials.technology/tutorials/19-how-to-do-a-regression-with-sklearn.html

We are going to choose fixed values of m and b for the formula y = mx + b,
where m is the angular coefficient and b is the intercept between the line and
the x axis. The version for machine learning is y = βX  + β[0], where X is a
matrix of predictors, β is a matrix of coefficients, and β[0] is a constant 
value called the bias. 

This is a basic example using 300 points to create a simple line with 
a static error of +/- 1. 
"""

import random # random.uniform
import matplotlib.pyplot as plt # plotting functions
from sklearn import linear_model # fit, coef_, intercept_, predict, score
import numpy as np # mean function

# There exists parameters β[0], β[1], and σ^2, such that for any fixed
#  variable x, the dependent variable is related to x through the model
#  equation f(x) = β[1] * X + β[0] + ε
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

def basicExample():
    # array of tuples
    values = []
    # ordinary least squares Linear Regression
    regr = linear_model.LinearRegression()
    
    # generate 300 random point that we will use to train a model
    for i in range(0, 300):
        # independent variable
        x = random.uniform(1, 1000)
        # dependent variable
        y = f(x)
        values.append((x, y))
        
    # split the values into two series instead of a list of tuples
    x, y = zip(*values)
    # max x value in 300 points
    max_x = max(x)
    # min x value in 300 points
    min_x = min(x)
    
    # training data
    # n_samples is all but 20 elements
    # n_features is 1
    train_data_X = list(map(lambda x: [x], list(x[:-20])))
    # n_targets is 1
    train_data_Y = list(y[:-20])
    
    # test data
    # n_samples is 20 elements
    # n_features is 1
    test_data_X = list(map(lambda x: [x], list(x[-20:])))
    # n_targets is 1
    test_data_Y = list(y[-20:])
    
    # feed the linear regression with the train data to obtain a model.
    regr.fit(train_data_X, train_data_Y)
    
    # Estimated coefficient for the regression line
    m = regr.coef_[0]
    # Independent term in the linear model
    b = regr.intercept_
    print('\ny = {0} * x + {1}'.format(m, b))
    
    # Estimated y values
    est = regr.predict(test_data_X)
    # actual Y values
    act = test_data_Y
    # Minimizes the SSE errors
    leastSquare = np.mean((act - est) ** 2)
    # the sum of the elements along the axis divided by the number of element
    print("Mean squared error: %.2f" % leastSquare)
    
    """
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    r^2 = 1 - (u/v)
    """
    r_square = regr.score(test_data_X, test_data_Y)
    # Scores are between 0 and 1, with a larger score indicating a better fit
    print('Variance score: %.2f' % r_square)
    
    # variance
    v = leastSquare / (len(train_data_X) - 2)
    print("Delta squared: %.2f" % v)
    
    # Display the training data points
    plt.scatter(x, y, color='blue')
    # Display the testing data points
    plt.scatter(test_data_X, test_data_Y, color='yellow')
    # Display the regression line 
    # x1 = min_x, x2 = max_x
    # y1 = b    , y2 = m * max_x + b
    # color is red
    plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.title('r = 1 Perfect Possitive Correlation', fontsize=16)
    plt.xlabel('x', fontsize=13)
    plt.ylabel('y', fontsize=13)
    

basicExample()
