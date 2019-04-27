import numpy as np 

def get_theta_and_parity(feature_col: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
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
    theta, parity = 0, 0

    for i in range(len(y)):
        e1 = s_p_list[i] + t_m - s_m_list[i]
        e2 = s_m_list[i] + t_p - s_p_list[i]
        if e1 < error:
            error = e1
            theta = feature_col[i]
            parity = -1
        elif e2 < error:
            error = e2
            theta = feature_col[i]
            parity = 1
    return (theta, parity)


def adaboost(data: list, T: int) -> dict:
    """
    This is an implementation of the AdaBoost algorithm mentioned in the 
    Viola-Jones paper.
    
    Arguments:
        data {List} -- Right now, a tuple of x & y
        T {int} -- Number of features needed 

    Returns:
        classifier_list -- List of weak classifiers
    """

    # For now, assuming that the x & y are packed into data
    x, y = data

    # Get number of samples
    n = len(y)
    m = len(y[y == 0])
    l = n - m
    w = np.ones_like(y, dtype=np.float32)
    w[y == 1] *= 1 / (2. * l)
    w[y == 0] *= 1 / (2. * m)
    w /= np.sum(w)

    error_so_far = 1e10

    cache = {}

    classifier_list = []
    
    for t in range(T):
        for i in range(x.shape[1]):
            feature_col = x[:, i]
            theta, parity = get_theta_and_parity(feature_col, y, w)
            error = 0.
            
            for j in range(x.shape[0]):
                if parity * theta > parity * feature_col[j]:
                    pred = 1
                else:
                    pred = 0
                
                error += np.abs(pred - y[j]) * w[j]

            if error < error_so_far:
                cache["theta"] = theta
                cache["parity"] = parity
                cache["error"] = error
                cache["index"] = j
                error_so_far = error

        beta = cache["error"] / (1 - cache["error"])
        alpha = np.log(1 / beta)

        parity, theta = cache["parity"], cache["theta"]
        for i in range(x.shape[0]):
            if parity * theta > parity * x[i, cache["index"]]:
                pred = 1
            else:
                pred = 0
            error = np.abs(pred - y)
            w[i] *= beta ** (1 - cache["error"])
        
        classifier_dict = {
            "theta": theta,
            "alpha": alpha,
            "parity": parity,
            "index": cache["index"],
            "weights": w
        }
        classifier_list.append(classifier_dict)
    return classifier_list



