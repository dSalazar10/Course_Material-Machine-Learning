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

Number of correct matches: 1273
Total number of data points: 1348
Ratio of correct predictions: 0.944362017804

Classification report
             precision    recall  f1-score   support

          0       1.00      0.97      0.99       139
          1       0.94      0.95      0.94       137
          2       0.98      0.98      0.98       129
          3       0.96      0.94      0.95       141
          4       0.97      0.95      0.96       127
          5       0.88      0.96      0.92       147
          6       0.95      0.97      0.96       135
          7       0.92      1.00      0.96       126
          8       0.93      0.84      0.88       134
          9       0.93      0.89      0.91       133

avg / total       0.95      0.94      0.94      1348

        Confusion matrix
[[135   0   0   0   1   3   0   0   0   0]
 [  0 130   1   0   0   3   1   1   0   1]
 [  0   0 126   0   0   0   0   2   1   0]
 [  0   0   0 132   0   3   0   1   3   2]
 [  0   2   0   0 121   0   0   2   0   2]
 [  0   1   0   0   1 141   1   1   0   2]
 [  0   0   0   0   0   2 131   0   2   0]
 [  0   0   0   0   0   0   0 126   0   0]
 [  0   4   1   4   1   4   5   1 112   2]
 [  0   2   0   2   1   4   0   3   2 119]]

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
    
    Penalty=None
    Number of correct matches: 1257
    Total number of data points: 1348
    Ratio of correct predictions: 0.932492581602
    
    Penalty='l2'
    Number of correct matches: 1221
    Total number of data points: 1348
    Ratio of correct predictions: 0.905786350148
    
    Penalty='l1'
    Number of correct matches: 1256
    Total number of data points: 1348
    Ratio of correct predictions: 0.93175074184
    
    Penalty: None (93.2%), 'l1' (93.1%), and 'l2' (90.5%)
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=5, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
    
    Shuffle: True
    Number of correct matches: 1267
    Total number of data points: 1348
    Ratio of correct predictions: 0.939910979228

    Shuffle: False
    Number of correct matches: 1225
    Total number of data points: 1348
    Ratio of correct predictions: 0.908753709199
    
    Shuffle=True
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=63, tol=None, shuffle=False, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
    
    Fit_intercept: True
    Number of correct matches: 1267
    Total number of data points: 1348
    Ratio of correct predictions: 0.939910979228
    
    Fit_intercept: False
    Number of correct matches: 1240
    Total number of data points: 1348
    Ratio of correct predictions: 0.919881305638
    
    fit_intercept=True
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=63, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)

    tol: 1.0
    Number of correct matches: 1233
    Total number of data points: 1348
    Ratio of correct predictions: 0.9146884273

    tol: None
    Number of correct matches: 1267
    Total number of data points: 1348
    Ratio of correct predictions: 0.939910979228

    tol=None
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=63, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
    
    
    class_weight: "balanced"
    Number of correct matches: 1262
    Total number of data points: 1348
    Ratio of correct predictions: 0.936201780415

    class_weight: None
    Number of correct matches: 1267
    Total number of data points: 1348
    Ratio of correct predictions: 0.939910979228
    
    class_weight=None
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=63, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
    """
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=56, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
    
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
    print "Total number of data points:", len(matches)
    # Ratio of correct predictions
    print "Ratio of correct predictions:", matches.sum() / float(len(matches))
    # Classification report
    print "\nClassification report\n", metrics.classification_report(act, est)
    # Confusion matrix
    print "\tConfusion matrix\n", metrics.confusion_matrix(act, est)
    
plot_data()
perceptron()
