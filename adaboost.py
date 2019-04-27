import numpy as np 

def get_epsilon_and_parity(feature_col: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
    """
    
    Arguments:
        feature_col {np.ndarray} -- A column of features 
        y {np.ndarray}
    
    Returns:
        tuple -- A tuple of epsilon and parity
    """
    # Flatten if necessary
    feature_col = feature_col.flatten()
    sorted_indices = np.argsort(feature_col)

    feature_col, y = feature_col[sorted_indices], y[sorted_indices]
    w = w[sorted_indices]
    
    # Referred from https://stackoverflow.com/questions/39109848/viola-jones-threshold-value-haar-features-error-value
    s_p, s_m, t_p, t_m = 0, 0, 0, 0

    s_p_list, s_m_list = [], []

    for i in range(len(y)):
        if y[i] == 0:
            s_m += w[i]
            t_m += w[i]
        else:
            s_p += w[i]
            t_m += w[i]
        s_m_list.append(s_m)
        s_p_list.append(s_p)
    
    error = 1e10
    epsilon, parity = 0, 0

    for i in range(len(y)):
        e1 = s_p_list[i] + t_m - s_m_list[i]
        e2 = s_m_list[i] + t_p - s_p_list[i]
        if e1 < error:
            error = e1
            epsilon = feature_col[i]
            parity = -1
        elif e2 < error:
            error = e2
            epsilon = feature_col[i]
            parity = 1
    return (epsilon, parity)



def adaboost(data: list, T: int):
    """
    This is an implementation of the AdaBoost algorithm mentioned in the 
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
    w[y == 1] /= (2 * l)
    w[y == 0] /= (2 * m)

    for t in range(T):
        w /= np.sum(w)


