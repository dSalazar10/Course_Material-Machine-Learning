from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

diabetes = load_diabetes()

lr = LinearRegression(normalize=True)
rg = Ridge(0.001, normalize=True)

lr_scores = cross_val_score(lr, diabetes.data, diabetes.target, cv=10)
print("LinearRegression:", lr_scores.mean())

rg_scores = cross_val_score(rg, diabetes.data, diabetes.target, cv=10)
print("Ridge:", rg_scores.mean())

# Ridge regression with built-in cross-validation
rg = RidgeCV(alphas=(1.0, 0.1, 0.01, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
rg.fit(diabetes.data, diabetes.target)
# Estimated regularization parameter.
print(rg.alpha_)
