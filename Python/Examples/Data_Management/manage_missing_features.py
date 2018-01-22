import numpy as np
from sklearn.preprocessing import Imputer

data = np.array([[1, np.nan, 2], [2, 3, np.nan], [-1, 4, 2]])
print("Original Data:\n", data)

imp = Imputer(strategy='mean')
print("\nMean:\n", imp.fit_transform(data))

imp = Imputer(strategy='median')
print("\nMedian:\n", imp.fit_transform(data))

imp = Imputer(strategy='most_frequent')
print("\nMost_frequent:\n", imp.fit_transform(data))
