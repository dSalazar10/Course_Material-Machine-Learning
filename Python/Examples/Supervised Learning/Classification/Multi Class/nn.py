import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class NeuralNetwork:
  def __init__(self, W=np.array(np.random.rand(2,1)), b=np.random.rand(1)[0], epoch=100, learn_rate=0.01,last_loss=None):
    self.w_ = W
    self.b_ = b
    self.e_ = epoch
    self.l_ = learn_rate
    self.ll_ = last_loss
    np.random.seed(143)

  def test_train_split(self, processed_data):
    sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
    return processed_data.iloc[sample], processed_data.drop(sample)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  def sigmoid_prime(self, x):
    return self.sigmoid(x) * (1 - self.sigmoid(x))
  # def error_formula(self, y, output):
  #   return - y*np.log(output) - (1 - y) * np.log(1-output)
  def error_term_formula(self, x, y, output):
    return (y - output) * self.sigmoid_prime(x)

  def fit(self, X, y):
    """
    This calculates the new boundary line based on the weight, input, and learning rate

    Inputs:
    * X: array of inputs
    * y: array of labels
    """
    for e in range(self.e_):
        del_w = np.zeros(self.w_.shape)
        for x, y in zip(X.values, y):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = self.sigmoid(np.dot(x, self.w_))

            # The error, the target minus the network output
            #error = self.error_formula(y, output)

            # The error term
            error_term = self.error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        self.w_ += self.l_ * del_w / X.shape[0]

  def predict(self, X):
    """
    This will provide the results. It calculates the Matrix
    Multiplication of the weights and the inputs and adds the
    biased. Then it uses the step method to predict the results.
    Inputs:
    * X: this is the matrix of inputs
    Output:
    * returns 1 if calculation is positive
    * returns 0 if calculation is negative
    """
    return self.sigmoid(np.dot(X, self.w_))

  def display(self, X, y):
    """
    Creates a scatter plot that includes both point groups, blue and red

    Input:
    * X: inputs
    * y: labels
    """
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')
    plt.title("Solution boundary")
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    m = -self.w_[0]/self.w_[1]
    x = np.arange(-10, 10, 0.1)
    b = -self.b_/self.w_[1]
    plt.plot(x, m*x+b, "black")

# Test usage
if 0:
  clf = NeuralNetwork()
  data = pd.read_csv('data.csv', header=None)
  X = np.array(data[[0,1]])
  y = np.array(data[2])
  np.random.seed(42)
  clf.fit(X, y)
  clf.predict(X)
  clf.display(X, y)
