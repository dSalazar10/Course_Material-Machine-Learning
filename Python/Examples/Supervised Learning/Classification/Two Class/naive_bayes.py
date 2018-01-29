"""
naive_bayes.py

Supervised Learning: Classification of Handwritten Digits
This tutorial will apply scikit-learn to the classification of handwritten digits
using a Gaussian Naive Bayes Classifier and the digits dataset. One thing that
is new is the ability to visualize the data using a Dimensionality Reduction 
technique.

Given these projections of the data, which numbers do you think a classifier 
might have trouble distinguishing?

According to the graph in plot_projection, the numbers 1 and 7 seem to be the 
troubling numbers for classification. 

It turns out Gaussian Naive Bayes is generally not sufficiently accurate 
for real-world data, but can perform surprisingly well, for instance on text data.

Why did we split the data into training and validation sets?
We split them into validation sets so that we could plug the training set into
the Gaussian Naive Bayes Classifier and to cross reference the results with the
training data to predict the accuracy of the classifier. In one test, we got an
83% match.

Our Classification report concludes our previous prediction, 1 had a 77% precision,
and 7 had a 78% precision. The ones I did not see was 2, which had a 79% precision,
and 8 which did the worst with a 56% precision.

According to our confusion matrix, 7 was confused for 4 the most and 8 was
confused for 3 the most.

http://www.scipy-lectures.org/packages/scikit-learn/#supervised-learning-classification-of-handwritten-digits
https://matplotlib.org/devdocs/gallery/images_contours_and_fields/interpolation_methods.html


Results:
Number of correct matches: 1112

Total number of data points: 1348

Ratio of correct predictions: 0.824925816024

Classification report
             precision    recall  f1-score   support

          0       1.00      0.96      0.98       128
          1       0.86      0.61      0.71       142
          2       0.79      0.81      0.80       132
          3       0.93      0.63      0.75       135
          4       0.96      0.82      0.88       143
          5       0.91      0.91      0.91       138
          6       0.97      0.96      0.97       139
          7       0.75      0.98      0.85       128
          8       0.51      0.89      0.64       132
          9       0.89      0.71      0.79       131

avg / total       0.86      0.82      0.83      1348

        Confusion matrix
[[123   1   0   0   4   0   0   0   0   0]
 [  0  86  11   0   0   2   1   1  39   2]
 [  0   0 107   1   1   0   2   0  20   1]
 [  0   1  10  85   0   4   0   5  28   2]
 [  0   1   0   0 117   2   0  19   4   0]
 [  0   0   0   3   0 125   0   4   5   1]
 [  0   2   0   0   0   2 134   0   1   0]
 [  0   0   0   0   0   0   0 125   1   2]
 [  0   4   3   0   0   2   0   3 117   3]
 [  0   5   4   2   0   1   1   9  16  93]]
"""


from matplotlib import pyplot as plt # figure, imshow, scatter, colorbar, show
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Digits dataset
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

# Analyze the data set, reduce it to 2 dimensions (x, y), and display the graph
def plot_projection():
    plt.figure()
    """
    Principal component analysis:
     Depending on shape (axis == 1 && size == 1), it uses either
     1) LAPACK implementation of the full SVD
     2) Randomized truncated SVD
     3) ARPACK implementation of the truncated SVD
    """
    pca = PCA(n_components=2)
    # Fit the model with X and apply the dimensionality reduction on X.
    proj = pca.fit_transform(digits.data)
    # Plot the model
    plt.scatter(proj[:, 0], proj[:, 1], c=digits.target)
    # Prints the color bar to the right of the plot
    # each color represents a number in the model
    # From (0:purple) to (9:yellow)
    plt.colorbar()


# Classify with Gaussian naive Bayes
def classification():
    # Create data for the  model
    # split the data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, 
                                                        digits.target, 
                                                        test_size = 0.75,
                                                        train_size = 0.25)
    
    # Create the model
    # Gaussian Naive Bayes Classifier class
    clf = GaussianNB()
    
    # Train the model
    # Fit Gaussian Naive Bayes according to X, y
    clf.fit(X_train, y_train)
    
    # estimated data
    est = clf.predict(X_test)
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
    print("\nNumber of correct matches:", matches.sum())
    # Count of data points
    print("\nTotal number of data points:", len(matches))
    # Ratio of correct predictions
    print("\nRatio of correct predictions:", matches.sum() / float(len(matches)))
    # Classification report
    print("\nClassification report\n", metrics.classification_report(act, est))
    # Confusion matrix
    print("\tConfusion matrix\n", metrics.confusion_matrix(act, est))
    plt.show()

plot_data()
plot_projection()
classification()
