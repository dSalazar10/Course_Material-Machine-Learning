"""
http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_boston, load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_regression

print("\nFeature selection that uses Variance Threshold\n")
X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
print(X)
selector = VarianceThreshold()
print(selector.fit_transform(X))

print("\nFeature selection that uses SelectKBest\n")
regr_data = load_boston()
print(regr_data.data.shape)

kb_regr = SelectKBest(f_regression)
X_b = kb_regr.fit_transform(regr_data.data, regr_data.target)
print(X_b.shape)
print(kb_regr.scores_)

print("\nFeature selection that uses SelectPercentile\n")
class_data = load_iris()
print(class_data.data.shape)

perc_class = SelectPercentile(chi2, percentile=15)
X_p = perc_class.fit_transform(class_data.data, class_data.target)
print(X_p.shape)
print(perc_class.scores_)
