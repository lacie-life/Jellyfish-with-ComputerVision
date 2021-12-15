import pytest
import numpy as np
import logging
from proj3_code.ransac import calculate_num_ransac_iterations, ransac_fundamental_matrix
from proj3_code import two_view_data
from proj3_code import fundamental_matrix
from proj3_code.feature_matching.utils import load_image, PIL_resize
from proj3_code import ransac


def test_calculate_num_ransac_iterations():
    Fail = False
    data_set = [(0.99, 1, 0.99, 1),
                (0.99, 10, 0.9, 11),
                (0.9, 15, 0.5, 75450),
                (0.95, 5, 0.66, 22)]

    for prob_success, sample_size, ind_prob, num_samples in data_set:
        S = calculate_num_ransac_iterations(
            prob_success,
            sample_size,
            ind_prob
        )
        assert pytest.approx(num_samples, abs=1.0) == S


def test_ransac_find_inliers():

    F = [[-2.49684084e-08, 7.37279178e-07, -5.99944364e-05],
              [-6.83245580e-07, -2.23634574e-08, 3.47641240e-03],
              [1.65302076e-04, -3.16334642e-03, -3.80986850e-02]]
    F = np.array(F)
    x_1s = np.load('../data/inliers2_a.npy')
    x_0s = np.load('../data/inliers2_b.npy')
    outliers = [1, 3, 4, 5, 10]
    for outlier in outliers:
        x_0s[outlier] += 3
    x_0s, x_1s = two_view_data.preprocess_data(x_0s, x_1s)

    inliers = ransac.find_inliers(x_0s, F, x_1s, 2)

    print(inliers)

    assert outliers not in inliers
    assert inliers.shape[0] == x_0s.shape[0] - len(outliers)


def test_ransac_fundamental_matrix_error():

    points_a = np.load('../unit_tests/pointsa.npy')
    points_b = np.load('../unit_tests/pointsb.npy')
    error_tolerance = 1

    F, inliers_x_0, inliers_x_1 = ransac_fundamental_matrix(points_a, points_b)

    x_0s, x_1s = two_view_data.preprocess_data(inliers_x_0, inliers_x_1)
    res = fundamental_matrix.signed_point_line_errors(x_0s, F, x_1s)
    res = np.abs(res)
    res = np.average(res)
    print('average residual = ', res)

    assert res < error_tolerance


def test_ransac_fundamental_matrix_fit():

    x_0s = np.load('../data/points2_a.npy')
    x_1s = np.load('../data/points2_b.npy')
    error_tolerance = 20.0

    F, inliers_x_0, inliers_x_1 = ransac_fundamental_matrix(x_0s, x_1s)

    x_0s, x_1s = two_view_data.preprocess_data(x_0s, x_1s)
    res = fundamental_matrix.signed_point_line_errors(x_0s, F, x_1s)
    res = np.abs(res)
    res = np.average(res)
    print('average residual = ', res)

    assert res < error_tolerance
