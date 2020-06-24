import numpy as np
import pandas as pd
class NeuralNetwork:
  d_ = True
  def __init__(self, W=np.array(0, dtype=float), b=0, epoch=1000, learn_rate=0.5):
    """
    This is the constructor class
    Input:
    * W: an array of weights
    * b: the bias
    * epoch: the number of loops to fit model
    * learn_rate: the scale of gradient descent steps
    Ouput:
    * returns a log-loss neural network object
    """
    self.w_ = W
    self.b_ = b
    self.e_ = epoch
    self.l_ = learn_rate
    np.random.seed(143)

  def test_train_split(self, data):
    """
    This will keep 90% of the data for training and 10% for testing.
    There are better ratios.

    Input:
    * data: labeled data (collection of features and targets)

    Output:
    * returns a tuple containing 2/3 training data and 1/3 testing data
    """
    # Get a random sample
    sample = np.random.choice(data.index, size=int(len(data)*0.75), replace=False)
    # train_data, test_data
    return data.iloc[sample], data.drop(sample)

  def sigmoid(self, x):
    # Return sigmoid(x)
    return 1 / (1 + np.exp(-x))

  def sigmoid_prime(self, x):
    # Return the derivitive of sigmoid(x)
    return self.sigmoid(x) * (1 - self.sigmoid(x))

  def error_term_formula(self, x, y, output):
    """
    output = ŷ = W_i * x_i
    y = 1 or 0
    """
    # 1 - output is negative if y is 0 and output is high
    # 1 - output is positive if y is 1 and output is high
    return (y - output) * self.sigmoid_prime(x)

  def fit(self, features, targets):
    """
    This calculates the new boundary line based on the weight, input, and learning rate

    Inputs:
    * X: array of features
    * y: array of targets
    """
    # Get num_targets
    n_records, n_features = features.shape
    # Update weights to a sample of the normal distribution (1/(sqrt(n_features)))
    self.w_ = np.random.normal(scale=1 / n_features**.5, size=n_features)
    # Repeat until stopping criterion is satisfied
    for _ in range(self.e_):
      # Create a gradient array: ΔW = [w_1, w_2, ..., w_n]
      del_w = np.zeros(self.w_.shape)
      # Choose one sample from training set
      for x, y in zip(features.values, targets):
        # Calculate the prediction sample
        output = self.sigmoid(np.dot(x, self.w_))
        # Calculate loss function for that prediction sample
        error_term = self.error_term_formula(x, y, output)
        # Calculate gradient from loss function
        del_w += error_term * x
      # Update model parameters based on gradient and learning rate
      self.w_ += self.l_ * del_w / n_records

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
    # Returns 1 / (1 + np.exp(-(Wx+b)))
    return self.sigmoid(np.dot(X, self.w_) + self.b_)

  def score(self, X_t, y_t):
    """
    This will find out how accurate the fitted model is using the test data

    Input:
    * X_t: the training features
    * y_t: the training targets
    """
    # Make a prediction using the test data
    predictions = self.predict(X_t) > 0.5
    # Calculate accuracy on test data
    accuracy = np.mean(predictions == y_t)
    return accuracy

  def one_hot_encoder(self, data, col, ax=1):
    # Make dummy variables for rank
    one_hot_data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=ax)
    # Drop the previous rank column
    return one_hot_data.drop(col, axis=1)

# Test usage
if 1:
  """
  1. Choose one sample from training set
  2. Calculate loss function for that single sample
  3. Calculate gradient from loss function
  4. Update model parameters a single step based on gradient and learning rate
  5. Repeat from 1) until stopping criterion is satisfied
  """
  clf = NeuralNetwork()

  # Pulling the data into a tableu
  data = pd.read_csv('student_data.csv')

  # Drill-down the rank column
  processed_data = clf.one_hot_encoder(data, "rank")

  # Scaling the columns
  processed_data['gre'] = processed_data['gre']/800
  processed_data['gpa'] = processed_data['gpa']/4.0

  # Split the data 2/3 train and 1/3 test
  train_data, test_data = clf.test_train_split(processed_data)
  
  # Splitting inputs and labels
  features = train_data.drop('admit', axis=1)
  targets = train_data['admit']
  features_test = test_data.drop('admit', axis=1)
  targets_test = test_data['admit']

  # Train the model on the training data
  clf.fit(features, targets)
  # Test the model on the testing data
  accuracy = clf.score(features_test, targets_test)
  print("Prediction accuracy: {:.3f}".format(accuracy))
