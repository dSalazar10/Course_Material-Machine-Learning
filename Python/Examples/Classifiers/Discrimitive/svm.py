"""
svm.py

An intro to linear classification using a Linear Support Vector Machine (SVM) 
which constructs a maximum-margin separating hyperplane between data classes 
in an n-dimensional space.

This will work well for practical problems such as document classification, 
and more generally for problems with many variables (features), reaching 
accuracy levels comparable to non-linear classifiers while taking less time to 
train and use. In Y. Yang, X. Liu, "A re-examination of text categorization", 
Proc. ACM SIGIR Conference, pp. 42–49, (1999)., it is stated that SVM was
introduced for solving two-class pattern recognition problems.

https://www.pyimagesearch.com/2016/08/22/an-intro-to-linear-classification-with-python/

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
   weights or parameters of our classifier that we’ll actually be optimizing.
   Based on the output of our score function and loss function, we’ll be 
   tweaking and fiddling with the values of our weight matrix to increase 
   classification accuracy.
   
There are two primary advantages to utilizing parameterized learning:

1.  Once we are done training our model, we can discard the input data and keep 
    only the weight matrix W and the bias vector b. This substantially reduces 
    the size of our model since we only need to store two sets of vectors 
    (versus the entire training set).
2.  Classifying new test data is fast. In order to perform a classification, 
    all we need to do is take the dot product of W and x_{i}, followed by 
    adding in the bias b. Doing this is substantially faster than needing 
    to compare each testing point to every training example.

"""

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np # array
import argparse # ArgumentParser class
import cv2 # cvtColor, calcHist, normalize, imread
import os # path.sep

# Accepts an input image, converts it to the HSV color space, and then
# computes a 3D color histogram using the supplied number of bins for each
# channel.
def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    #handle normalizing the histogram
    hist = cv2.normalize(hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()

# Parse the path to our input Kaggle Dogs vs. Cats dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())
 
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the data matrix and labels list
data = []
labels = []

# loop over the input imagePaths
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	# extract a color histogram from the image, then update the
	# data matrix and labels list
	hist = extract_color_histogram(image)
	data.append(hist)
	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        
# The labels  list is represented as a list of strings, either “dog” or “cat”
# So we encode the labels, converting them from strings to integers (1 or 0)
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.25, random_state=42)
 
# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)
 
# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions,
	target_names=le.classes_))
