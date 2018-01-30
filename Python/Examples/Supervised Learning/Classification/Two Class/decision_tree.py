"""
decision_tree.py

Results:
    
Number of correct matches: 896

Total number of data points: 1348

Ratio of correct predictions: 0.6646884273

Classification report
             precision    recall  f1-score   support

          0       0.85      0.77      0.81       139
          1       0.66      0.68      0.67       137
          2       0.78      0.70      0.73       129
          3       0.61      0.68      0.64       141
          4       0.67      0.71      0.69       127
          5       0.68      0.39      0.49       147
          6       0.80      0.85      0.83       135
          7       0.62      0.79      0.70       126
          8       0.41      0.37      0.39       134
          9       0.59      0.74      0.66       133

avg / total       0.67      0.66      0.66      1348

        Confusion matrix
[[107   0   1   2   5   2   4   3   7   8]
 [  3  93   4   6   2   1   5   6  11   6]
 [  3  16  90   2   4   5   1   4   3   1]
 [  0   5   5  96   2   3   0  11   8  11]
 [  4   5   1   5  90   0   6   9   2   5]
 [  5   1   1  12  13  57   9  11  25  13]
 [  1   2   3   2   3   3 115   0   6   0]
 [  0   2   1   3   6   2   1 100   2   9]
 [  1  11  10  21  10  10   2   4  50  15]
 [  2   5   0   8   0   1   0  12   7  98]]

"""
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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
        
def dtree():
    # split the data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size = 0.75,
                                                    train_size = 0.25, 
                                                    random_state=0)
    # Create the model
    # Decision Tree Classifier.
    dtree = DecisionTreeRegressor(random_state=0)
    
    # Train the model
    # Fit dtree according to X, y
    dtree.fit(X_train, y_train)
    
    # estimated data
    est = dtree.predict(X_test)
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
dtree()
