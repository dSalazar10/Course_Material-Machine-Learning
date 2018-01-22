from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()

pca = PCA(n_components=36, whiten=True)
X_pca = pca.fit_transform(digits.data / 255)
print(pca.explained_variance_ratio_)

X_rebuilt = pca.inverse_transform(X_pca)
print(X_rebuilt)
