import cv2
import numpy as np

from collections import Counter

class box:
    def __init__(self, x: int, y: int, l: int, b: int):
        """
        A box in the integral image, given start points and dimensions.

        Arguments:
            x {int} -- x co-ordinate of origin
            y {int} -- y co-ordinate of origin
            l {int} -- Length of box
            b {int} -- Width of box
        """
        self.x = x
        self.y = y
        self.l = l
        self.b = b

    def __str__(self):
        return "Box: \n\t(x, y):" +  "(" + str(self.x) + "," + str(self.y) + ")" + "\n\t(l, b):" + "(" + str(self.l) + "," + str(self.b) + ")"

def integral_image(img: np.ndarray) -> np.ndarray:
    """
    This function returns the integral image for a given image.

    Referred - https://stackoverflow.com/questions/25557973/efficient-summed-area-table-calculation-with-numpy

    Arguments:
        img {np.ndarray} -- The image for which the integral image needs to
        be computed for

    Returns:
        integral_image {np.ndarray} -- The integral image
    """
    if len(img.shape) == 2:
        m, n = img.shape
        integral_image = np.zeros((m + 1, n + 1))
        integral_image[1:, 1:] = img.cumsum(0).cumsum(1)
        return integral_image

    # Get image shape
    m, n, o = img.shape

    # Declare zeros array for initial condition
    integral_image = np.zeros((m + 1, n + 1, o))

    # Do cumulative summation
    integral_image[1:, 1:, :] = img.cumsum(0).cumsum(1)

    return integral_image

def get_integral_sum_box(integral_img: np.ndarray, box: box) -> float:
    """
    Same as get_integral_sum but with box

    Arguments:
        integral_image {np.ndarray} -- The integral image
        box {box} -- box object

    Returns:
        feature -- Same as get_integral_sum
    """
    return get_integral_sum(integral_img, box.x, box.y, box.b, box.l)


def get_integral_sum(integral_img: np.ndarray, x: int, y: int, h: int, w: int) -> float:
    """
    Gets summation of a table in the integral image

    Arguments:
        integral_img {np.ndarray} -- The integral image
        x {int} -- X co-ordinate of the rectangle
        y {int} -- Y co-ordinate of the rectangle
        w {int} -- Width of the rectangle
        h {int} -- Height of the rectangle

    Returns:
        feature -- Difference in co-ords
    """
    feature = integral_img[y + h][x + w] + integral_img[y][x] - \
        integral_img[y + h][x] - integral_img[y][x + w]

    try:
        _ = iter(feature)
        return np.sqrt(np.sum(feature ** 2))
    except:
        return feature

def get_features(img: np.ndarray) -> list:
    """
    This computes the features, given the rectangular boxes

    Arguments:
        img {np.ndarray} -- Input image

    Returns:
        features {List} -- List of features computed
    """
    integral_img = integral_image(img)

    shape = integral_img.shape
    m, n = shape[0], shape[1]

    features = []

    feature_shape = [[2, 1],
                     [1, 2],
                     [3, 1],
                     [1, 3],
                     [2, 2]]

    feature_dict = {}

    count = 0
    for f in feature_shape:
        x, y = f[0], f[1]
        for i in range(m - x + 1):
            for j in range(n - y + 1):
                for k in range(x, m - i, x):
                    for l in range(y, n - j, y):
                        if f == [2, 1]:
                            box1 = box(i, j, k // 2, l)
                            box2 = box(i + k // 2, j, k // 2, l)
                            feature_dict[count] = [f, [box1, box2]]

                            box1 = get_integral_sum_box(integral_img, box1)
                            box2 = get_integral_sum_box(integral_img, box2)

                            features.append(box1 - box2)

                        if f == [1, 2]:
                            box1 = box(i, j, k, l // 2)
                            box2 = box(i, j + l // 2, k, l // 2)
                            feature_dict[count] = [f, [box1, box2]]

                            box1 = get_integral_sum_box(integral_img, box1)
                            box2 = get_integral_sum_box(integral_img, box2)

                            features.append(box2 - box1)

                        if f == [3, 1]:
                            box1 = box(i, j, k // 3, l)
                            box2 = box(i + k // 3, j, k // 3, l)
                            box3 = box(i + 2 * k // 3, j, k // 3, l)

                            feature_dict[count] = [f, [box1, box2, box3]]

                            box1 = get_integral_sum_box(integral_img, box1)
                            box2 = get_integral_sum_box(integral_img, box2)
                            box3 = get_integral_sum_box(integral_img, box3)

                            features.append(box2 - box1 - box3)

                        if f == [1, 3]:
                            box1 = box(i, j, k, l // 3)
                            box2 = box(i, j + l // 3 , k, l // 3)
                            box3 = box(i, j + 2 * l // 3, k, l // 3)
                            feature_dict[count] = [f, [box1, box2, box3]]

                            box1 = get_integral_sum_box(integral_img, box1)
                            box2 = get_integral_sum_box(integral_img, box2)
                            box3 = get_integral_sum_box(integral_img, box3)

                            features.append(box2 - box1 - box3)

                        if f == [2, 2]:
                            box1 = box(i, j, k // 2, l // 2)
                            box2 = box(i + k // 2, j, k // 2, l // 2)
                            box3 = box(i, j + l // 2, k // 2, l // 2)
                            box4 = box(i + k // 2, j + l // 2, k // 2, l // 2)

                            feature_dict[count] = [f, [box1, box2, box3, box4]]

                            box1 = get_integral_sum_box(integral_img, box1)
                            box2 = get_integral_sum_box(integral_img, box2)
                            box3 = get_integral_sum_box(integral_img, box3)
                            box4 = get_integral_sum_box(integral_img, box4)

                            features.append(box1 - box2 - box3 + box4)
                        count += 1
    return features, feature_dict
