"""
svm.py

An intro to linear classification using a Linear Support Vector Machine (SVM) 
which constructs a maximum-margin separating hyperplane between data classes 
in an n-dimensional space.

This will work well for practical problems such as document classification, 
and more generally for problems with many variables (features), reaching 
accuracy levels comparable to non-linear classifiers while taking less time to 
train and use. In Y. Yang, X. Liu, "A re-examination of text categorization", 
Proc. ACM SIGIR Conference, pp. 42-49, (1999)., it is stated that SVM was
introduced for solving two-class pattern recognition problems.

parameterization is the process of defining the necessary parameters of a 
given model, defining our problem in terms of:
1. Data: This is our input data that we are going to learn from. This data 
    includes both the data points (e.x., feature vectors, color histograms, 
    raw pixel intensities, etc.) and their associated class labels.
    The training dataset is x[i] and the feature vector is y[i], where
    i = 1 ... N and y[i] = 1 ... K. We have N data points of dimensionality D,
    separated into K unique categories. 
    For example, if we have 25,000 images of cats and dogs, N = 25000, then each 
    image is characterized by a 3D color histogram with 8 bins per channel, 
    respectively. The feature vector is D = 8 x 8 x 8 = 512 entries. Since 
    there are only two types of images, Cat or Dog, K = 2.
2. Score function: A function that accepts our data as input and maps the data
    to class labels. For instance, given our input feature vectors, the score 
    function takes these data points, applies some function f 
    (our score function), and then returns the predicted class labels.
    The linear mapping function is f(x[i], W, b) = W * x[i] + b, where x[i] is
    a single column vector with shape [D x 1], the weight matrix W has the shape
    [K x D], and the bias vector b has the shape [K x 1], where K is the number 
    of class labels and D is the dimensionality of the feature vector. the only 
    parameters that we have any control over are our weight matrix W and our 
    bias vector b; x[i] and y[i] do not change.
3. Loss function: A loss function quantifies how well our predicted class 
    labels agree with our ground-truth labels. The higher level of agreement 
    between these two sets of labels, the lower our loss (and higher our 
    classification accuracy, at least on the training data). Our goal is to 
    minimize our loss function, thereby increasing our classification accuracy.
4. Weight matrix: The weight matrix, typically denoted as W, is called the 
    weights or parameters of our classifier that we'll actually be optimizing.
    Based on the output of our score function and loss function, we'll be 
    tweaking and fiddling with the values of our weight matrix to increase 
    classification accuracy.

There are two primary advantages to utilizing parameterized learning:

1. Once we are done training our model, we can discard the input data and keep 
    only the weight matrix W and the bias vector b. This substantially reduces 
    the size of our model since we only need to store two sets of vectors 
    (versus the entire training set).
2. Classifying new test data is fast. In order to perform a classification, 
    all we need to do is take the dot product of W and x_{i}, followed by 
    adding in the bias b. Doing this is substantially faster than needing 
    to compare each testing point to every training example.

    
Results:

Number of correct matches: 418

Total number of data points: 450

Ratio of correct predictions: 0.928888888889

Classification report
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        37
          1       0.83      0.93      0.88        43
          2       0.93      0.91      0.92        44
          3       0.83      0.96      0.89        45
          4       0.93      0.97      0.95        38
          5       0.98      0.98      0.98        48
          6       0.93      0.98      0.95        52
          7       0.98      0.94      0.96        48
          8       1.00      0.71      0.83        48
          9       0.94      0.94      0.94        47

avg / total       0.93      0.93      0.93       450

        Confusion matrix
[[37  0  0  0  0  0  0  0  0  0]
 [ 0 40  0  0  0  0  3  0  0  0]
 [ 0  1 40  3  0  0  0  0  0  0]
 [ 0  0  1 43  0  0  0  0  0  1]
 [ 0  0  0  1 37  0  0  0  0  0]
 [ 0  1  0  0  0 47  0  0  0  0]
 [ 0  1  0  0  0  0 51  0  0  0]
 [ 0  1  0  1  1  0  0 45  0  0]
 [ 0  4  2  3  1  0  1  1 34  2]
 [ 0  0  0  1  1  1  0  0  0 44]]
"""
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
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
        
def svm():
    # split the data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size = 0.25, 
                                                    random_state=0)
    # Create the model
    # Linear Support Vector Classifier.
    svm = LinearSVC()
    
    # Train the model
    # Fit SVM according to X, y
    svm.fit(X_train, y_train)
    
    # estimated data
    est = svm.predict(X_test)
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
svm()
