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

    feature_col, y = feature_col.copy()[sorted_indices], y.copy()[sorted_indices]
    w = w.copy()[sorted_indices]

    # Referred from https://stackoverflow.com/questions/39109848/viola-jones-threshold-value-haar-features-error-value
    # and https://stackoverflow.com/questions/9777282/the-best-way-to-calculate-the-best-threshold-with-p-viola-m-jones-framework
    
    total_pos_sum, total_neg_sum = 0, 0
    running_pos_sum, running_neg_sum = 0, 0
    running_pos_sum_list, running_neg_sum_list = [0] * len(y), [0] * len(y)

    for i in range(len(y)):
        if y[i] == 0:
            running_neg_sum += w[i]
            total_neg_sum += w[i]
        else:
            running_pos_sum += w[i]
            total_pos_sum += w[i]
        running_neg_sum_list[i] = running_neg_sum
        running_pos_sum_list[i] = running_pos_sum

    error = 1e10
    theta, parity = 0, 0

    for i in range(len(y)):
        error_neg = running_pos_sum_list[i] + total_neg_sum - running_neg_sum_list[i]
        error_pos = running_neg_sum_list[i] + total_pos_sum - running_pos_sum_list[i]
        if error_neg < error:
            error = error_neg
            theta, parity = feature_col[i], -1
        elif error_pos < error:
            error = error_pos
            theta, parity = feature_col[i], 1
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
    w[y == 1] *= 1 / (2. * l + 1)
    w[y == 0] *= 1 / (2. * m + 1)


    classifier_list = []
    t = 0
    
    def can_continue(i, classifier_list):
        for wc in classifier_list:
            if wc["index"] == i:
                return False
        return True

    print("Building", T, "classifiers..")
    while t <= T:
        print ("at t =", t)
        cache = {}
        error_so_far = 1e10
        w /= np.sum(w)
        for i in range(x.shape[1]):
            if not can_continue(i, classifier_list):
                continue
            feature_col = x[:, i]
            theta, parity = get_theta_and_parity(feature_col, y, w)

            parity_arr = (parity * theta > parity * feature_col) * 1.
            error = np.sum(np.abs(parity_arr - y) * w)
            
            if error < error_so_far:
                cache["theta"] = theta
                cache["parity"] = parity
                cache["error"] = error
                cache["index"] = i
                error_so_far = error

        beta = cache["error"] / (1 - cache["error"])
        alpha = np.log(1 / beta)

        parity, theta = cache["parity"], cache["theta"]


        pred = parity * theta > parity * x[:, cache["index"]]
        pred = pred * 1.

        error = np.abs(pred - y)
        w = w * np.power(beta, 1 - error)

        classifier_dict = {
            "theta": theta,
            "alpha": alpha,
            "parity": parity,
            "index": cache["index"],
        }
        print(classifier_dict, classifier_list)
        t += 1
        classifier_list.append(classifier_dict)
    
    print(classifier_list)
    return classifier_list
