import numpy as np 

def adaboost(data: List, T: int):
    """This is an implementation of the AdaBoost algorithm mentioned in the 
    Viola-Jones paper.
    
    Arguments:
        data {List} -- Right now, a tuple of x & y
        T {int} -- Number of features needed 

    Returns:
        weights -- Weights for the strongest classifer
    """

    # For now, assuming that the x & y are packed into data
    x, y = data

    # Get number of samples
    n = len(y)
    m = len(y[y == 0])
    l = n - m
    
    w = np.ones_like(y)
    w[y == 1] /= l
    w[y == 0] /= m

    for t in range(T):
        w /= np.sum(w)
        sorted_indices = np.argsort(x[i])
        features_sorted = x[i][sorted_indices]

