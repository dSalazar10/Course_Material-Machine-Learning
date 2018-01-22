
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

boston = load_boston()
print("Boston Data's Shape:", boston.data.shape)
# split the data
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.1)
# set up the Linear Regression Class
lr = LinearRegression(normalize=True)
# Train the model
lr.fit(X_train, Y_train)
# Check the accuracy of a regression
print("Score = ", lr.score(X_test, Y_test))

# Evaluate a score by cross-validation
scores1 = cross_val_score(lr, boston.data, boston.target, cv=7, scoring='neg_mean_squared_error')
scores2 = cross_val_score(lr, boston.data, boston.target, cv=10, scoring='r2')

print("\nneg_mean_squared_error:\n", scores1)
print("r2:\n", scores2)

print("\nneg_mean_squared_error average = ", scores1.mean())
print("r2 average = ", scores2.mean())

print("\nneg_mean_squared_error std = ", scores1.std())
print("r2 std = ", scores2.std())

print('\ny = ' + str(lr.intercept_) + ' ')
for i, c in enumerate(lr.coef_):
    print(str(c) + ' * x' + str(i))
    

"""
import matplotlib.pyplot as plt # plotting functions

# Display graphs for each feature
for n in range(1,13):
    plt.figure(n)
    plt.plot(boston.data[:,n])
"""
