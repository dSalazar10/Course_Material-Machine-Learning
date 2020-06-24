import pandas as pd
import numpy as np
class Utilities:
  def one_hot_encoder(self, data, col, ax=1):
    """
    This will convert a single row of targets that are sequential and 
    convert it into the appropriate number of colums with a prefix
    matching the column's original name
    For example 'rank' into ['rank_1','rank_2',...,'rank_n']

    Input:
    * data: labeled data (collection of features and targets)
    * col: the string label of the column to modify
    * ax: the axis of modification
    """
    # Make dummy variables for rank
    one_hot_data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=ax)
    # Drop the previous rank column
    return one_hot_data.drop(col, axis=1)

  def test_train_split(self, data):
    """
    This will keep 75% of the data for training and 25% for testing.

    Input:
    * data: labeled data (collection of features and targets)

    Output:
    * returns a tuple containing 2/3 training data and 1/3 testing data
    """
    # Get a random sample
    sample = np.random.choice(data.index, size=int(len(data)*0.75), replace=False)
    # train_data, test_data
    return data.iloc[sample], data.drop(sample)
