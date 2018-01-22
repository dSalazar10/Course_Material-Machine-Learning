import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# very small dataset made of 10 categorical samples with two features each
X = np.random.uniform(0.0, 1.0, size=(10, 2))
# cannot immediately be processed by any algorithm
Y = np.random.choice(('Male','Female'), size=(10))
print(X[0])
print(Y[0])

# used to normalize labels
le = LabelEncoder()
# associating to each category label a progressive integer number
yt = le.fit_transform(Y)
print(yt)

# The inverse transformation can be obtained with the following
# create a new dataset
output = [1, 0, 1, 1, 0, 0]
# use the classes to create a new Y dataset
decoded_output = [le.classes_[i] for i in output]
print(decoded_output)

# Problem: a classifier which works with real values will then consider 
# similar numbers according to their distance, without any concern for 
# semantics. So we use one-hot encoding, which binarizes the data.
lb = LabelBinarizer()
Yb = lb.fit_transform(Y)
print(Yb)
print(lb.inverse_transform(Yb))
