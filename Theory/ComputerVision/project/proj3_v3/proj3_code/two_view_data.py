"""Data from two views."""

import numpy as np

def preprocess_data(x_0s, x_1s):
    """Preprocess data to make them homogeneous."""
    x_0s = np.append(x_0s, np.ones((x_0s.shape[0], 1)), axis=1)
    x_1s = np.append(x_1s, np.ones((x_1s.shape[0], 1)), axis=1)
    return x_0s, x_1s

def read_from_text():
    x_0s = np.loadtxt('../data/pic_p_pts.txt')
    x_1s = np.loadtxt('../data/pic_p_pts.txt')
    x_0s, x_1s = preprocess_data(x_0s, x_1s)
    return x_0s, x_1s

