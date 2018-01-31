"""
perceptron_max_iter.py

The program finds the best input from 1 to 1000 for max_iter in the Perceptron() class.
Results show that 56 is the optimal number for the digits dataset.

1-100:    56 0.944362017804
101-200:  56 0.944362017804
201-300:  56 0.944362017804
301-400:  56 0.944362017804
401-500:  56 0.944362017804
501-600:  56 0.944362017804
601-700:  56 0.944362017804
701-800:  56 0.944362017804
801-900:  56 0.944362017804
901-1000: 56 0.944362017804

"""
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

# handwritten digits data set
from sklearn.datasets import load_digits
digits = load_digits()

def predict(X_train, X_test, y_train, y_test, n):
    
    perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=n, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
    perceptron.fit(X_train, y_train)
    matches = (perceptron.predict(X_test) == y_test)
    return (matches.sum() / float(len(matches)))

def perceptron():
    # split the data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size = 0.75,
                                                    train_size = 0.25, 
                                                    random_state=0)
    max_result = 0.0
    iter_n = 0
    for n in range(1,1000):
        result = predict(X_train, X_test, y_train, y_test, n)
        if result > max_result:
            max_result = result
            iter_n = n
        if n % 100 == 0:
            print iter_n, max_result
    print iter_n, max_result
    
perceptron()
