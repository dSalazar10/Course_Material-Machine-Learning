import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LinearRegression:
  def __init__(
      self, 
      W=np.array(np.random.rand(2,1)), 
      b=np.random.rand(1)[0], 
      epoch=100, 
      learn_rate=0.01
    ):
    self.w_ = W
    self.b_ = b
    self.e_ = epoch
    self.l_ = learn_rate
    np.random.seed(143)

  def test_train_split(self, processed_data):
    """
    The size of the testing set will be 10% of the total data.

    Input:
    * processed_data: one-hot encoded multi-class data set
    Output:
    * returns train_data, test_data
    """
    sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
    train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
     

  def fit(self, X, y):
    """
    This calculates the new boundary line based on the weight, input, and learning rate

    Inputs:
    * X: array of inputs
    * y: array of labels
    """
    self.b_ += max(X.T[0])
    for e in range(self.e_):
      # for each element in the input array
      for i in range(len(X)):
          # Calculates the step function of the results of the input
          y_hat = self.predict(X[i])
          # If prediction = 0
          if y[i]-y_hat == 1:
              # Change weight to weight + α * input
              self.w_[0] += X[i][0]*self.l_
              self.w_[1] += X[i][1]*self.l_
              # Change b to b + α
              self.b_ += self.l_
          # If prediction = 1
          elif y[i]-y_hat == -1:
              # Change weight to weight - α * input
              self.w_[0] -= X[i][0]*self.l_
              self.w_[1] -= X[i][1]*self.l_
              # Change b to b - α
              self.b_ -= self.l_

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
    return 1 if ( ((np.matmul(X, self.w_) + self.b_)[0]) >= 0 ) else 0

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
  clf = LinearRegression()
  data = pd.read_csv('data.csv', header=None)
  X = np.array(data[[0,1]])
  y = np.array(data[2])
  np.random.seed(42)
  clf.fit(X, y)
  clf.predict(X)
  clf.display(X, y)

