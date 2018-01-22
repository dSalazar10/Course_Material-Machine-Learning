"""
http://scikit-learn.org/stable/modules/decomposition.html#nmf

An alternative approach to decomposition that assumes that the data and the
components are non-negative
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import NMF

iris = load_iris()
print("Iris data:", iris.data.shape)

# Non-Negative Matrix Factorization (NMF)
nmf = NMF(n_components=3, init='random', l1_ratio=0.1)
Xt = nmf.fit_transform(iris.data)
print("Reconstruction error:", nmf.reconstruction_err_)

print("First feature of iris data:", iris.data[0])
print("First feature of transofrmed data:", Xt[0])
print("Inverse of NMF", nmf.inverse_transform(Xt[0]))
