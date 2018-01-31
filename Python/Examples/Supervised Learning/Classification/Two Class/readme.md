Useful for predicting categories with two classes.

SVM: 
- greater than 100 features, linear model

averaged perceptron: 
- Fast training, linear model

logistic regression: 
- Fast training, linear model

Bayes point machine: 
- Fast training, linear model

decision forest: 
- Accuracy, fast training

boosted decision tree: 
- Accuracy, fast training, large memory footprint

decision jungle: 
- accuracy, small memory footprint

locally deep SVM: 
- greater than 100 features

nueral network: 
- accuracy, long training times

Todo:
- decision forest
- boosted decision tree
- decision jungle
- locally deep SVM
- nueral network
- comment Perceptron
- comment Decision Tree

Performance review of linear modeling of the digits dataset

Naive Bayes Classfication
- Number of correct matches: 1112
- Total number of data points: 1348
- Ratio of correct predictions: 0.824925816024

Logistic Regression Classfication:
- Number of correct matches: 1288
- Total number of data points: 1348
- Ratio of correct predictions: 0.955489614243

Linear SVM Classification:
- Number of correct matches: 1259
- Total number of data points: 1348
- Ratio of correct predictions: 0.933976261128

Default Perceptron Classification:
perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=5, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
- Number of correct matches: 1257
- Total number of data points: 1348
- Ratio of correct predictions: 0.932492581602

Tuned Perceptron Classification:
perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                            max_iter=63, tol=None, shuffle=True, verbose=0,
                            eta0=1.0, n_jobs=1, random_state=0,
                            class_weight=None, warm_start=False)
- Number of correct matches: 1267
- Total number of data points: 1348
- Ratio of correct predictions: 0.939910979228

Decision Tree Classification:
- Number of correct matches: 896
- Total number of data points: 1348
- Ratio of correct predictions: 0.6646884273
