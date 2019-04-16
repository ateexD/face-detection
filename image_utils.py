import cv2
import numpy as np


def integral_image(img: np.ndarray) -> np.ndarray:
    """This function returns the integral image for a given image.

    Referred - https://stackoverflow.com/questions/25557973/efficient-summed-area-table-calculation-with-numpy

    Arguments:
        img {np.ndarray} -- The image for which the integral image needs to 
        be computed for

    Returns:
        integral_image {np.ndarray} -- The integral image
    """
    # Get image shape
    m, n, o = img.shape

    # Declare zeros array for initial condition
    integral_image = np.zeros((m + 1, n + 1, o))

    # Do cumulative summation
    integral_image[1:, 1:, :] = img.cumsum(0).cumsum(1)

    return integral_image

def get_integral_sum(integral_img: np.ndarray, x: int, y: int, h: int, w: int) -> float:
    """Gets summation of a table in the integral image

    Arguments:
        integral_img {np.ndarray} -- The integral image
        x {int} -- X co-ordinate of the rectangle
        y {int} -- Y co-ordinate of the rectangle
        w {int} -- Width of the rectangle
        h {int} -- Height of the rectangle

    Returns:
        float -- [description]
    """
    feature = integral_img[y + h][x + w] + integral_img[y][x] - \
        integral_img[y + h][x] - integral_img[y][x + w]
    return feature

def get_features(img: np.ndarray) -> List:
    """This computes the features, given the rectangular boxes

    Arguments:
        img {np.ndarray} -- Input image 

    Returns:
        features {List} -- List of features computed
    """
    integral_img = integral_image(img)
    
    m, n, o = integral_img.shape

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            for k in range(0, m, i):
                for l in range(0, n, j):
                    pass
    # TODO - implement this
    