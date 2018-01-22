from sklearn.decomposition import SparsePCA
from sklearn.datasets import load_digits
digits = load_digits()

spca = SparsePCA(n_components=60, alpha=0.1)
X_spca = spca.fit_transform(digits.data / 255)

print(spca.components_.shape)
