from sklearn.feature_extraction import DictVectorizer, FeatureHasher

data = [
   { 'feature_1': 10.0, 'feature_2': 15.0 },
   { 'feature_1': -5.0, 'feature_3': 22.0 },
   { 'feature_3': -2.0, 'feature_4': 10.0 }
]

# Transforms lists of feature-value mappings to vectors
dv = DictVectorizer()
Y_dict = dv.fit_transform(data)
print(Y_dict.todense())
print(dv.vocabulary_)

# turns sequences of symbolic feature names into matrices
fh = FeatureHasher()
Y_hashed = fh.fit_transform(data)
print(Y_hashed.todense())
