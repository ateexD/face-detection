import cv2
import numpy as np


def integral_image(img: np.ndarray) -> np.ndarray:
    """This function returns the integral image
    for a given image.

    Referred - https://stackoverflow.com/questions/25557973/efficient-summed-area-table-calculation-with-numpy

    Arguments:
        img {np.ndarray} -- The image for which
        the integral image needs to be computed for.

    Returns:
        integral_image {np.ndarray} -- The integral
        image. 
    """
    # Get image shape
    m, n, o = img.shape

    # Declare zeros array for initial condition
    integral_image = np.zeros((m + 1, n + 1, o))

    # Do cumulative summation
    integral_image[1:, 1:, :] = img.cumsum(0).cumsum(1)

    return integral_image

        

