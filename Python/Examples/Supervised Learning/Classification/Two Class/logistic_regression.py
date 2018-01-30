#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
logistic_regression.py

Logistic regression is know as logit or MaxEnt.
solver parameter for Two-Class LogisticRegression():
    Small datasets: liblinear
    Large datasets: sag/saga

The goal of logistic regression is to find the best fitting model to describe
 the relationship between the dichotomous characteristic of interest 
 (dependent variable = response or outcome variable) and a set of independent
 (predictor or explanatory) variables. Logistic regression generates the 
 coefficients (and its standard errors and significance levels) of a formula
 to predict a logit transformation of the probability of presence of the 
 characteristic of interest: logit(p) = b0 + b1 * x1 + b2 * x2 + ... + bk * xk,
 where p is the probability of presence of the characteristic of interest. 
 
 The logit transformation is defined as the logged odds: 
     odds = p/(1-p) = (prob. of presence)/(prob. of absence), and 
     logit(p) = ln(p/(1-p))

 Rather than choosing parameters that minimize the sum of squared errors 
 (like in ordinary regression), estimation in logistic regression chooses
 parameters that maximize the likelihood of observing the sample values.
 https://www.medcalc.org/manual/logistic_regression.php
 
 
 Results:
     
Number of correct matches: 1288

Total number of data points: 1348

Ratio of correct predictions: 0.955489614243

Classification report
             precision    recall  f1-score   support

          0       0.99      0.99      0.99       139
          1       0.89      0.97      0.93       137
          2       0.99      0.98      0.99       129
          3       0.95      0.94      0.94       141
          4       0.98      0.95      0.97       127
          5       0.97      0.95      0.96       147
          6       0.97      0.99      0.98       135
          7       0.96      1.00      0.98       126
          8       0.90      0.87      0.89       134
          9       0.95      0.92      0.93       133

avg / total       0.96      0.96      0.96      1348

        Confusion matrix
[[138   0   0   0   0   1   0   0   0   0]
 [  0 133   0   0   0   0   1   0   3   0]
 [  0   0 127   0   0   0   0   0   2   0]
 [  0   1   0 132   0   2   0   1   3   2]
 [  1   3   0   0 121   0   0   0   0   2]
 [  0   1   0   1   1 139   1   2   1   1]
 [  0   0   0   0   1   0 133   0   1   0]
 [  0   0   0   0   0   0   0 126   0   0]
 [  0   8   1   3   0   1   2   1 117   1]
 [  0   4   0   3   0   0   0   1   3 122]]
 
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# handwritten digits data set
from sklearn.datasets import load_digits
digits = load_digits()

# Display the data that will be used, represented as multiple subplots in a 
# grid format. Each subplot will contain the string representation of the 
# image's number.
def plot_data(n = 64):
    # size of image: a width x height tuple in inches
    fig = plt.figure(figsize=(6, 6))
    # add space at the right and top of each image 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # add n sub_plots
    for i in range(n):
        # nrows, ncols, index
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        # indexed image in black/white/grey with no blur between pixels
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # x = 0 y = 7 (origin is top left corner)
        ax.text(0, 7, str(digits.target[i]))
        
def logistic_regression():
    # split the data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    train_size = 0.25, 
                                                    random_state=0)
    # Create the model
    # Logistic Regression (aka logit, MaxEnt) classifier.
    logit = LogisticRegression()
    
    # Train the model
    # Fit Logistic Regression according to X, y
    logit.fit(X_train, y_train)
    
    # estimated data
    est = logit.predict(X_test)
    # actual data
    act = y_test
    
    # Display the results
    # size of image: a width x height tuple in inches
    fig = plt.figure(figsize=(6, 6))
    # add space at the right and top of each image 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # add n sub_plots
    for i in range(64):
        # nrows, ncols, index
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        # indexed results in black/white/grey with no blur between pixels
        ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
              interpolation='nearest')
        # label the image with the target value
        if est[i] == act[i]: # correct
            ax.text(0, 7, str(est[i]), color='green')
        else: # error
            ax.text(0, 7, str(est[i]), color='red')

    # Quantify the performance
    # Correct matches
    matches = (est == act)
    print "\nNumber of correct matches:", matches.sum()
    # Count of data points
    print "\nTotal number of data points:", len(matches)
    # Ratio of correct predictions
    print "\nRatio of correct predictions:", matches.sum() / float(len(matches))
    # Classification report
    print "\nClassification report\n", metrics.classification_report(act, est)
    # Confusion matrix
    print "\tConfusion matrix\n", metrics.confusion_matrix(act, est)
    
plot_data()
logistic_regression()
