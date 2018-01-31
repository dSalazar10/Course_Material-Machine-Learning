"""
perceptron.py

The Perceptron is another simple algorithm suitable for large scale learning. 
By default:
- It does not require a learning rate.
- It is not regularized (penalized).
- It updates its model only on mistakes.

The last characteristic implies that the Perceptron is slightly faster to 
train than SGD with the hinge loss and that the resulting models are sparser.

https://en.wikipedia.org/wiki/Perceptron

o(x1, x2, ..., xN) = ((w0 + w1x1 + w2x2 + ... + wNxN) > 0) ? (1) : (-1)  



Results:

Number of correct matches: 1257

Total number of data points: 1348

Ratio of correct predictions: 0.932492581602

Classification report
             precision    recall  f1-score   support

          0       0.99      0.97      0.98       139
          1       0.90      0.88      0.89       137
          2       0.99      0.99      0.99       129
          3       0.99      0.84      0.91       141
          4       0.96      0.95      0.96       127
          5       0.87      0.96      0.91       147
          6       0.96      0.96      0.96       135
          7       0.98      0.97      0.97       126
          8       0.79      0.91      0.84       134
          9       0.94      0.89      0.92       133

avg / total       0.94      0.93      0.93      1348

        Confusion matrix
[[135   0   0   0   1   3   0   0   0   0]
 [  0 121   0   0   0   3   1   0  11   1]
 [  0   0 128   0   0   0   0   0   1   0]
 [  0   4   1 119   0   4   0   2   7   4]
 [  0   1   0   0 121   0   0   0   5   0]
 [  1   2   0   0   1 141   1   0   0   1]
 [  0   3   0   0   0   2 129   0   1   0]
 [  0   0   0   0   2   1   0 122   1   0]
 [  0   3   0   0   1   4   3   0 122   1]
 [  0   1   0   1   0   4   0   1   7 119]]

"""
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
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
        
def perceptron():
    # split the data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size = 0.75,
                                                    train_size = 0.25, 
                                                    random_state=0)
    # Create the model
    # Perceptron Classifier.
    """
    penalty : None, 'l2' or 'l1' or 'elasticnet'
    - Regularization term; Default is None
    
    alpha : float
    - Regularization Coefficient; Defaults to 0.0001 if regularization is used
    
    fit_intercept : bool
    - Estimation; Defaults to True
    
    max_iter : int, optional
    - Max Epochs; Defaults to 5
    
    tol : float or None, optional
    - Stopping criterion; Defaults to None
    
    shuffle : bool, optional, default True
    - Should shuffle after epoch
    
    verbose : integer, optional
    - Verbosity level
    
    eta0 : double
    - Update Coefficient; Defaults to 1
    
    n_jobs : integer, optional
    - Number of CPUs for OVA; Defaults to 1
    
    random_state : int, RandomState instance or None, optional, default None
    - seed of the Shuffle pseudo random number generator
    
    class_weight : dict, {class_label: weight} or "balanced" or None, optional
    - Weights associated with classes; Defaults to 1
    
    warm_start : bool, optional
    - If True, reuse previous call's solution as init, else, erase solution
    
    Perceptron() is equivalent to 
    SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant",
                  penalty=None)
    
    """
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=5, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False, n_iter=None)
    
    # Train the model
    # Fit linear model with Stochastic Gradient Descent
    perceptron.fit(X_train, y_train)
    
    # estimated data
    est = perceptron.predict(X_test)
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
perceptron()
