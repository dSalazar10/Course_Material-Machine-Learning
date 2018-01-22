"""
Kernel PCA is a powerful instrument when we think of our dataset as made up of
elements that can be a function of components (in particular, radial-basis or
polynomials) but we aren't able to determine a linear relationship among them.

http://scikit-learn.org/stable/modules/metrics.html#linear-kernel                                            
"""

from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt # plotting functions

Xb, Yb = make_circles(n_samples=500, factor=0.1, noise=0.05)
kpca = KernelPCA(n_components=2,
                 kernel='rbf',
                 fit_inverse_transform=True,
                 gamma=1.0)
X_kpca = kpca.fit_transform(Xb)

plt.figure(1)
plt.scatter(Xb[:,0], Xb[:,1], color='blue')
plt.title('Circle with a blob inside', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)

# The plot shows a separation just like expected, and it's also possible to
# see that the points belonging to the central blob have a curve distribution
# because they are more sensitive to the distance from the center.
plt.figure(2)
plt.scatter(X_kpca[:,0], X_kpca[:,1], color='blue')
plt.title('Projection of dataset into new space', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)

