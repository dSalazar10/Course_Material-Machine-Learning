from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

# set the seed for reproducability
rs = check_random_state(1000)
# load necessary dataset
boston = load_boston()

# split dataset into traiing and test sets
X_train, X_test, Y_train, Y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.25,
                                                    random_state=rs)
