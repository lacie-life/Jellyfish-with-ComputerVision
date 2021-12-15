"""Unit tests for fundamental_matrix module."""

import math
import unittest

import numpy as np

from proj3_code import least_squares_fundamental_matrix
from proj3_code import two_view_data
from proj3_code.fundamental_matrix import (create_F, 
                                point_line_distance,
                                signed_point_line_errors)

K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float)

def verify(function) -> str:
  """ Will indicate with a print statement whether assertions passed or failed
    within function argument call.

    Args:
    - function: Python function object

    Returns:
    - string
  """
  try:
    function()
    return "\x1b[32m\"Correct\"\x1b[0m"
  except AssertionError:
    return "\x1b[31m\"Wrong\"\x1b[0m"

class TestFundamentalMatrix(unittest.TestCase):

    def setUp(self):
        """Create F for testing, corresponding to stereo rig."""
        # Create calibration for a 640*480 image
        self.F = create_F(K, np.eye(3), [1, 0, 0])
        self.x_0s = [[100, 200, 1],
                   [100, 140, 1],
                   [200, 240, 1],
                   [300, 340, 1],
                   [200, 440, 1],
                   [100, 480, 1],
                   [300, 340, 1],
                   [300, 440, 1],
                   [500, 100, 1]]
        self.x_1s = [[300, 200, 1],
                   [100, 140, 1],
                   [200, 240, 1],
                   [300, 340, 1],
                   [200, 440, 1],
                   [100, 480, 1],
                   [600, 340, 1],
                   [600, 440, 1],
                   [100, 100, 1]]

    def assertEquivalent(self, x1, x2):
        """Assert two vectors are equivelent up to scale."""
        np.testing.assert_array_almost_equal(x1/float(x1[-1]), x2/float(x2[-1]))

    def test_mapping(self):
        """Make sure mapping is correct. Not super-string for stereo."""
        self.assertEquivalent(
            np.array([0, -1, 200]), np.dot(self.F.T, [100, 200, 1]))
        self.assertEquivalent(
            np.array([0, 1, -200]), np.dot([100, 200, 1], self.F))

    def test_least_squares_optimize(self):
        """Test optimize with LM, needs 9 points."""
        p0 = self.F.flatten()
        result = least_squares_fundamental_matrix.optimize(
            p0, np.array(self.x_0s), np.array(self.x_1s))
        F = np.reshape(result, (3, 3))
        np.testing.assert_array_almost_equal(F, self.F)

    def test_signed_point_line_errors(self):
        """ Check line-point errors."""
        errors = signed_point_line_errors(self.x_0s, self.F, self.x_1s)
        self.assertEqual(errors, [0.0]*18)

    def test_point_line_distance(self):
        """
        Testing point line distance calculation
        """
        line = [3, 4, 6]
        point = [2, 1, 1]
        distance = 16.0/5

        test_distance = point_line_distance(line, point)
        self.assertEqual(distance, test_distance)

    def test_point_line_distance_zero(self):
        line = [3, 3, -6]
        point = [1, 1, 1]

        test_distance = point_line_distance(line, point)
        self.assertEqual(test_distance, 0)

class TestFundamentalMatrix2(unittest.TestCase):
    """Second example with synthetic 3D example."""

    def setUp(self):
        """Create F for testing, corresponding to stereo rig."""
        # Create 3D points in a synthetic scene
        z = 4
        points = [[0, 0, z, 1],
                  [-1, -1, z-1, 1],
                  [-1, -1, z+1, 1],
                  [-1, 1, z-1, 1],
                  [-1, 1, z+1, 1],
                  [1, -1, z-1, 1],
                  [1, -1, z+1, 1],
                  [1, 1, z-1, 1],
                  [1, 1, z+1, 1]]
        # project
        P1 = np.hstack((K, np.reshape([0, 0, 0], (3, 1))))
        self.x_1s = [np.dot(P1, p) for p in points]
        self.x_1s = [x_1/x_1[2] for x_1 in self.x_1s]
        # Set up a second camera to the left, looking theta degrees to right
        theta = math.radians(10)
        R = np.vstack(([math.cos(theta), 0, -math.sin(theta)],
                       [0, 1, 0],
                       [math.sin(theta), 0, math.cos(theta)]))
        t = np.reshape([-1, 0, 0], (3, 1))
        P2 = np.dot(K, np.hstack((R, t)))
        self.x_0s = [np.dot(P2, p) for p in points]
        self.x_0s = [x_0/x_0[2] for x_0 in self.x_0s]
        self.F = create_F(K, R, t)

    def test_least_squares_optimize(self):
        """Test optimize with LM, needs 9 points."""
        p0 = self.F.flatten()
        result = least_squares_fundamental_matrix.optimize(
            p0, np.array(self.x_0s), np.array(self.x_1s))
        F = np.reshape(result, (3, 3))
        np.testing.assert_array_almost_equal(F, self.F)

    def test_signed_point_line_errors(self):
        """ Check line-point errors."""
        errors = signed_point_line_errors(self.x_0s, self.F, self.x_1s)
        np.testing.assert_array_almost_equal(errors, [0.0]*18)

class TestFundamentalMatrix3(unittest.TestCase):
    """Second example with real example."""

    def setUp(self):
        """Create F for testing, corresponding to real data."""
        # Create 3D points in test scene
        self.x_0s = [[211,184,1],[190,257,1],[231,319,1], [198,397,1],[181,569,1],
                    [316,513,1], [113,270,1],[140,213,1],[244,180,1],[158,319,1]]
        self.x_1s = [[194.0,31,1],[106,154,1],[137,286,1], [58,401,1],[52,581,1],
                    [254,536,1], [24,150,1],[103,66,1],[248,34,1],[57,245,1]]
        self.F = [[ 6.41288932e-04, -3.09650307e-03,  1.14020370e-01],
                  [ 3.30242172e-03,  1.05145300e-03, -1.77170831e+00],
                  [-9.45095842e-01,  1.01150632e+00,  3.24039002e+02]]
        self.F = np.array(self.F)

    def test_least_squares_optimize(self):
        """Test optimize with LM, needs 9 points."""
        p0 = self.F.flatten()
        result = least_squares_fundamental_matrix.optimize(
            p0, np.array(self.x_0s), np.array(self.x_1s))
        F = np.reshape(result, (3, 3))
        F = F/10.0
        self_F = self.F/10.0
        np.testing.assert_array_almost_equal(F, self_F, decimal=0)

    def test_signed_point_line_errors(self):
        """Check line-point errors."""
        errors = signed_point_line_errors(self.x_0s, self.F, self.x_1s)

        actual_errors = [-0.010742, -0.020253, -0.110908, -0.2068  ,  0.8076  ,  1.453266,
                         -0.638929, -1.066478,  0.431201,  0.55922 , -0.193313, -0.249138,
                          0.536447,  0.891095, -0.065045, -0.10857 , -0.07294 , -0.139982,
                         -0.711676, -1.175722]
        np.testing.assert_array_almost_equal(errors, actual_errors, decimal=1)


if __name__ == '__main__':
    unittest.main()
