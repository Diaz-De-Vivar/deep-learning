import numpy as np


def argmax(iterable):
    """
    Returns the index of the maximum value in the iterable.

    Args:
        iterable: A sequence of comparable values

    Returns:
        The index of the maximum value
    """
    # Lambda function to find the index of the maximum value in the iterable
    # The lambda function is used to extract the second element of each tuple (the value) for comparison
    # and the first element (the index) is returned as the result. The second element is the value of the iterable at that index.
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def argmax_numpy(iterable):
    """
    Returns the index of the maximum value in the iterable.

    Args:
        iterable: A sequence of comparable values

    Returns:
        The index of the maximum value
    """
    # return np.argmax(iterable)
    return int(np.argmax(iterable)) # np.argmax returns a numpy int64 type, so convert to int