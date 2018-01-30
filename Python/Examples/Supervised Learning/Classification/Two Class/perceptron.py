"""
perceptron.py

The Perceptron is another simple algorithm suitable for large scale learning. 
By default:
- It does not require a learning rate.
- It is not regularized (penalized).
- It updates its model only on mistakes.

The last characteristic implies that the Perceptron is slightly faster to 
train than SGD with the hinge loss and that the resulting models are sparser.


Results:

Number of correct matches: 1253

Total number of data points: 1348

Ratio of correct predictions: 0.929525222552

Classification report
             precision    recall  f1-score   support

          0       1.00      0.97      0.99       139
          1       0.92      0.94      0.93       137
          2       0.98      0.92      0.95       129
          3       0.94      0.94      0.94       141
          4       0.98      0.94      0.96       127
          5       0.90      0.96      0.93       147
          6       0.94      0.96      0.95       135
          7       0.92      1.00      0.96       126
          8       0.80      0.80      0.80       134
          9       0.93      0.86      0.90       133

avg / total       0.93      0.93      0.93      1348

        Confusion matrix
[[135   0   0   0   0   3   0   0   1   0]
 [  0 129   1   0   0   1   1   1   3   1]
 [  0   1 119   0   0   0   0   2   7   0]
 [  0   0   0 132   0   3   0   1   3   2]
 [  0   3   0   0 119   0   0   2   2   1]
 [  0   1   0   0   1 141   1   1   0   2]
 [  0   0   0   0   0   2 130   0   3   0]
 [  0   0   0   0   0   0   0 126   0   0]
 [  0   5   1   6   1   4   7   1 107   2]
 [  0   1   0   2   1   3   0   3   8 115]]

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
    perceptron = Perceptron(max_iter=1000)
    
    # Train the model
    # Fit perceptron according to X, y
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
