from sklearn.preprocessing import OneHotEncoder

# the first feature is a binary index which indicates 'Male' or 'Female'
# the second feature is a count
data = [
   [0, 10],
   [1, 11],
   [1, 8],
   [0, 12],
   [0, 15]
]

# filter the dataset
oh = OneHotEncoder(categorical_features=[0])
Y_oh = oh.fit_transform(data)
print(Y_oh.todense())
