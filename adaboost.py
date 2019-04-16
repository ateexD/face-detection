def adaboost(data: List):
    """This is an implementation of the AdaBoost algorithm mentioned in the 
    Viola-Jones paper.
    
    Arguments:
        data {List} -- Right now, just a list of features.

    Returns:
        weights -- Weights for the strongest classifer
    """

    # For now, assuming that the x & y are packed into data
    x, y = data

    # TODO - Finish this
