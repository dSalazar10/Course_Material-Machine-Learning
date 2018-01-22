from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score
from scipy import sparse

diabetes = load_diabetes()

lr = LinearRegression(normalize=True)
rg = Ridge(0.001, normalize=True)
ls = Lasso(alpha=0.001, normalize=True)
en = ElasticNet(alpha=0.001, l1_ratio=0.8, normalize=True)

lr_scores = cross_val_score(lr, diabetes.data, diabetes.target, cv=10)
print("LinearRegression:", lr_scores.mean())

rg_scores = cross_val_score(rg, diabetes.data, diabetes.target, cv=10)
print("Ridge:", rg_scores.mean())

ls_scores = cross_val_score(ls, diabetes.data, diabetes.target, cv=10)
print("Lasso:", ls_scores.mean())

en_scores = cross_val_score(en, diabetes.data, diabetes.target, cv=10)
print("ElasticNet:", en_scores.mean())

# Built-in cross validation:

# RidgeCV
rg = RidgeCV(alphas=(1.0, 0.1, 0.01, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
rg.fit(diabetes.data, diabetes.target)
# Estimated regularization parameter.
print(rg.alpha_)

# LassoCV
ls = Lasso(alpha=0.001, normalize=True)
print(ls.fit(sparse.coo_matrix(diabetes.data), diabetes.target))

# ElasticNetCV
encv = ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8), normalize=True)
print(encv.fit(diabetes.data, diabetes.target))
print(encv.alpha_)
print(encv.l1_ratio_)

