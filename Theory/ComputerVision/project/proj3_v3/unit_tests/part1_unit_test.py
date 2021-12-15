#!/usr/bin/python3

import numpy as np

from proj3_code.projection_matrix import (
    projection,
    objective_func,
    decompose_camera_matrix,
    calculate_camera_center,
    estimate_camera_matrix
)

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


def test_projection():
    '''
        tests whether projection was implemented correctly
    '''

    test_3D = np.array([[311.49450897, 307.88750897,  28.83350897],
                        [306.29458458, 312.14758458,  30.85458458],
                        [307.69425074, 312.35825074,  30.41825074],
                        [308.91388726, 305.95088726,  28.06288726]])

    test_2D = np.array([[363.17462352, 359.53935346],
                        [356.48789847, 362.30774149],
                        [358.19257325, 362.8436046 ],
                        [361.14927848, 358.14751343]])

    dummy_matrix = np.array([[150,   0, 250, 120],
                             [  0, 150, 250, 120],
                             [  0,   0,   1, 120]])
    
    projected_2D = projection(dummy_matrix, test_3D)

    assert projected_2D.shape == test_2D.shape    
    assert np.allclose(projected_2D, test_2D, atol=1e-8)

def test_objective_func():
    '''
        tests whether the objective function has been implemented correctly
        by comparing fixed inputs and expected outputs
    '''
    
    test_input = np.array([ -8.53194648, -10.51316603,  14.36357683,   7.29274269,
                           -15.98865902,  -6.2424195 ,   1.19262228,  -2.63932353,
                            11.58506293,  17.16781278,  13.87057014])
    
    pts2d_path = '../data/pts2d-pic_b.txt'
    pts3d_path = '../data/pts3d.txt'

    points_2d = np.loadtxt(pts2d_path)
    points_3d = np.loadtxt(pts3d_path)

    kwargs = {'pts2d':points_2d,
              'pts3d':points_3d}

    test_output1 = np.array([ 731.58606112,  238.73773889,   22.58423066,  248.73012447,
			     204.58447632,  230.73110084,  903.58738896,  342.73797823,
			     635.588051  ,  316.73749425,  867.5842564 ,  177.73685444,
			     958.58836175,  572.73653067,  328.58523902,  244.73254193,
			     426.58697305,  386.73373087, 1064.58800996,  470.73758343,
			     480.58778955,  495.7341862 ,  964.5872493 ,  419.73668571,
			     695.5880278 ,  374.73678667,  505.58724   ,  372.73458746,
			     645.58778382,  452.73531122,  692.588577  ,  359.7377652 ,
			     712.58750082,  444.73546011,  465.58587339,  263.73417285,
			     591.5884491 ,  324.73738013,  447.58506859,  213.73418182]) 
    
    test_output2 = np.array([ -731.58606112,  -238.73773889,   -22.58423066,  -248.73012447,
		    	   -204.58447632,  -230.73110084,  -903.58738896,  -342.73797823,
			   -635.588051  ,  -316.73749425,  -867.5842564 ,  -177.73685444,
			   -958.58836175,  -572.73653067,  -328.58523902,  -244.73254193,
			   -426.58697305,  -386.73373087, -1064.58800996,  -470.73758343,
			   -480.58778955,  -495.7341862 ,  -964.5872493 ,  -419.73668571,
			   -695.5880278 ,  -374.73678667,  -505.58724   ,  -372.73458746,
			   -645.58778382,  -452.73531122,  -692.588577  ,  -359.7377652 ,
			   -712.58750082,  -444.73546011,  -465.58587339,  -263.73417285,
			   -591.5884491 ,  -324.73738013,  -447.58506859,  -213.73418182]) 

    output = objective_func(test_input, **kwargs)
    
    assert output.shape == test_output1.shape or output.shape == test_output2.shape

    assert np.allclose(np.sum(output), np.sum(test_output1), atol=1e-8) or \
	   np.allclose(np.sum(output), np.sum(test_output2), atol=1e-8)
   

def test_decompose_camera_matrix():
    '''
        tests whether projection was implemented correctly
    '''
    
    test_input = np.array([[ 122.43413524,  -58.4445669 ,   -8.71785439, 1637.28675475],
                           [   4.54429487,    3.30940264, -134.40907701, 2880.869899  ],
                           [   0.02429085,    0.02388273,   -0.01160657,    1.        ]])

    test_R = np.array([[ 0.70259051, -0.71156708,  0.00623408],
                        [-0.22535139, -0.23080127, -0.94654505],
                        [ 0.67496913,  0.66362871, -0.32251142]])

    test_K = np.array([[127.55394371,  -5.84978098,  46.66537651],
                       [  0.        , 125.43636838,  48.61193508],
                       [  0.        ,   0.        ,   0.03598809]])

    K, R = decompose_camera_matrix(test_input)
    

    assert K.shape == test_K.shape and R.shape == test_R.shape

    assert np.allclose(test_R, R, atol=1e-8)
    assert np.allclose(test_K, K, atol=1e-8)

def test_calculate_camera_center():
    '''
        tests whether projection was implemented correctly
    '''
    
    test_R = np.array([[ 0.70259051, -0.71156708,  0.00623408],
                        [-0.22535139, -0.23080127, -0.94654505],
                        [ 0.67496913,  0.66362871, -0.32251142]])

    test_K = np.array([[127.55394371,  -5.84978098,  46.66537651],
                       [  0.        , 125.43636838,  48.61193508],
                       [  0.        ,   0.        ,   0.03598809]])
    
    test_input = np.array([[ 122.43413524,  -58.4445669 ,   -8.71785439, 1637.28675475],
                           [   4.54429487,    3.30940264, -134.40907701, 2880.869899  ],
                           [   0.02429085,    0.02388273,   -0.01160657,    1.        ]])

    test_cc = np.array([-18.27559442, -13.32677465,  20.48757872])

    cc = calculate_camera_center(test_input, test_K, test_R)

    assert cc.shape == test_cc.shape

    assert np.allclose(test_cc, cc, atol=1e-8)


def test_estimate_camera_matrix():
    '''
        tests whether camera matrix estimation is done correctly
        given an initial guess
    '''

    initial_guess_K = np.array([[ 500,   0, 535],
                                [   0, 500, 390],
                                [   0,   0,  -1]])

    initial_guess_R = np.array([[ 0.5,   -1,  0],
                                [   0,    0, -1],
                                [   1,  0.5,  0]])

    initial_guess_I_t = np.array([[   1,    0, 0, 300],
                                  [   0,    1, 0, 300],
                                  [   0,    0, 1,  30]])

    initial_guess_P = np.matmul(initial_guess_K, np.matmul(initial_guess_R, initial_guess_I_t))
    pts2d_path = '../data/pts2d-pic_b.txt'
    pts3d_path = '../data/pts3d.txt'

    points_2d = np.loadtxt(pts2d_path)
    points_3d = np.loadtxt(pts3d_path)

    test_P_row = np.array([ -0.45582852, -0.30414814, 2.14988425, 166.18819427])
    
    P = estimate_camera_matrix(points_2d, points_3d, initial_guess_P)

    assert np.allclose(P[1,:], test_P_row, atol=1e-8)



