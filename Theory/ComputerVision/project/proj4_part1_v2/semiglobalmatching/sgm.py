"""
This code has been adapted from https://github.com/beaupreda/semi-global-matching/blob/master/sgm.py

python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

original author: David-Alexandre Beaupre
date: 2019/07/12
"""

import argparse
import sys
import time as t
import torch

import numpy as np

from proj4_code.disparity_map import calculate_cost_volume


class Direction:
  def __init__(self, direction=(0, 0), name='invalid'):
    """
    represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
    :param direction: (x, y) for cardinal direction.
    :param name: common name of said direction.
    """
    self.direction = direction
    self.name = name


# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
  def __init__(self):
    """
    represent the relation between the directions.
    """
    self.paths = [N, NE, E, SE, S, SW, W, NW]
    self.size = len(self.paths)
    self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
  def __init__(self, max_disparity=64, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):
    """
    represent all parameters used in the sgm algorithm.
    :param max_disparity: maximum distance between the same pixel in both images.
    :param P1: penalty for disparity difference = 1
    :param P2: penalty for disparity difference > 1
    :param csize: size of the kernel for the census transform.
    :param bsize: size of the kernel for blurring the images and median filtering.
    """
    self.max_disparity = max_disparity
    self.P1 = P1
    self.P2 = P2
    self.csize = csize
    self.bsize = bsize


def get_indices(offset, dim, direction, height):
  """
  for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
  :param offset: difference with the main diagonal of the cost volume.
  :param dim: number of elements along the path.
  :param direction: current aggregation direction.
  :param height: H of the cost volume.
  :return: arrays for the y (H dimension) and x (W dimension) indices.
  """
  y_indices = []
  x_indices = []

  for i in range(0, dim):
    if direction == SE.direction:
      if offset < 0:
        y_indices.append(-offset + i)
        x_indices.append(0 + i)
      else:
        y_indices.append(0 + i)
        x_indices.append(offset + i)

    if direction == SW.direction:
      if offset < 0:
        y_indices.append(height + offset - i)
        x_indices.append(0 + i)
      else:
        y_indices.append(height - i)
        x_indices.append(offset + i)

  return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset, parameters):
  """
  part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
  given direction)
  :param slice: M x D array from the cost volume.
  :param offset: ignore the pixels on the border.
  :param parameters: structure containing parameters of the algorithm.
  :return: M x D array of the minimum costs for a given slice in a given direction.
  """
  other_dim = slice.shape[0]
  disparity_dim = slice.shape[1]

  disparities = [d for d in range(disparity_dim)] * disparity_dim
  disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

  penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=np.float32)
  penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
  penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

  minimum_cost_path = np.zeros(
      shape=(other_dim, disparity_dim), dtype=np.float32)
  minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

  for i in range(offset, other_dim):
    previous_cost = minimum_cost_path[i - 1, :]
    current_cost = slice[i, :]
    costs = np.repeat(previous_cost, repeats=disparity_dim,
                      axis=0).reshape(disparity_dim, disparity_dim)
    costs = np.amin(costs + penalties, axis=0)
    minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
  return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths):
  """
  second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
  :param cost_volume: array containing the matching costs.
  :param parameters: structure containing parameters of the algorithm.
  :param paths: structure containing all directions in which to aggregate costs.
  :return: H x W x D x N array of matching cost for all defined directions.
  """
  height = cost_volume.shape[0]
  width = cost_volume.shape[1]
  disparities = cost_volume.shape[2]
  start = -(height - 1)
  end = width - 1

  aggregation_volume = np.zeros(
      shape=(height, width, disparities, paths.size), dtype=np.float32)

  path_id = 0
  for path in paths.effective_paths:
    print('\tProcessing paths {} and {}...'.format(
        path[0].name, path[1].name), end='')
    sys.stdout.flush()
    dawn = t.time()

    main_aggregation = np.zeros(
        shape=(height, width, disparities), dtype=np.float32)
    opposite_aggregation = np.copy(main_aggregation)

    main = path[0]
    if main.direction == S.direction:
      for x in range(0, width):
        south = cost_volume[0:height, x, :]
        north = np.flip(south, axis=0)
        main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
        opposite_aggregation[:, x, :] = np.flip(
            get_path_cost(north, 1, parameters), axis=0)

    if main.direction == E.direction:
      for y in range(0, height):
        east = cost_volume[y, 0:width, :]
        west = np.flip(east, axis=0)
        main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
        opposite_aggregation[y, :, :] = np.flip(
            get_path_cost(west, 1, parameters), axis=0)

    if main.direction == SE.direction:
      for offset in range(start, end):
        south_east = cost_volume.diagonal(offset=offset).T
        north_west = np.flip(south_east, axis=0)
        dim = south_east.shape[0]
        y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
        y_nw_idx = np.flip(y_se_idx, axis=0)
        x_nw_idx = np.flip(x_se_idx, axis=0)
        main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(
            south_east, 1, parameters)
        opposite_aggregation[y_nw_idx, x_nw_idx,
                             :] = get_path_cost(north_west, 1, parameters)

    if main.direction == SW.direction:
      for offset in range(start, end):
        south_west = np.flipud(cost_volume).diagonal(offset=offset).T
        north_east = np.flip(south_west, axis=0)
        dim = south_west.shape[0]
        y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
        y_ne_idx = np.flip(y_sw_idx, axis=0)
        x_ne_idx = np.flip(x_sw_idx, axis=0)
        main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(
            south_west, 1, parameters)
        opposite_aggregation[y_ne_idx, x_ne_idx,
                             :] = get_path_cost(north_east, 1, parameters)

    aggregation_volume[:, :, :, path_id] = main_aggregation
    aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
    path_id = path_id + 2

    dusk = t.time()
    print('\t(done in {:.2f} s)'.format(dusk - dawn))

  return aggregation_volume


def compute_costs(left_img, right_img, max_disparity, sim_fn, block_size=9, save_images=False):
  """
  first step of the sgm algorithm, matching cost based on census transform and hamming distance.
  :param left: left image.
  :param right: right image.
  :param parameters: structure containing parameters of the algorithm.
  :param save_images: whether to save census images or not.
  :return: H x W x D array with the matching costs.
  """

  return calculate_cost_volume(torch.FloatTensor(left_img), torch.FloatTensor(
      right_img), max_disparity, sim_fn, block_size=block_size
  ).numpy()/block_size**2


def select_disparity(aggregation_volume):
  """
  last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
  :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
  :return: disparity image.
  """
  volume = np.sum(aggregation_volume, axis=3)
  disparity_map = np.argmin(volume, axis=2)
  return disparity_map


def normalize(volume, parameters):
  """
  transforms values from the range (0, 64) to (0, 255).
  :param volume: n dimension array to normalize.
  :param parameters: structure containing parameters of the algorithm.
  :return: normalized array.
  """
  return 255.0 * volume / parameters.max_disparity


def sgm(im_left, im_right, output_name, max_disparity, sim_fn, block_size=9, save_images=False):
  """
  main function applying the semi-global matching algorithm.
  :return: void.
  """

  disparity = int(max_disparity)

  dawn = t.time()

  # parser = argparse.ArgumentParser()
  # parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
  # parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
  # parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
  # parser.add_argument('--disp', default=64, help='maximum disparity for the stereo pair')
  # parser.add_argument('--images', default=False, help='save intermediate representations (e.g. census images)')
  # args = parser.parse_args()

  parameters = Parameters(max_disparity=disparity,
                          P1=8.0/255, P2=128.0/255, csize=(7, 7), bsize=(3, 3))
  paths = Paths()

  # print('\nLoading images...')
  # left, right = load_images(left_name, right_name, parameters)
  left = im_left
  right = im_right

  print('\nStarting cost computation...')
  cost_volume = compute_costs(
      left, right, max_disparity, sim_fn, block_size, save_images)
  # if save_images:
  #     disparity_map = np.uint8(normalize(np.argmin(cost_volume, axis=2), parameters))
  #     cv2.imwrite('disp_map_cost_volume.png', disparity_map)

  print('\nStarting aggregation computation...')
  aggregation_volume = aggregate_costs(cost_volume, parameters, paths)

  print('\nSelecting best disparities...')
  disparity_map = np.float32(select_disparity(aggregation_volume))
  print('\nDone')
  return disparity_map
  # if save_images:
  #     cv2.imwrite('disp_map_no_post_processing.png', disparity_map)

  # print('\nApplying median filter...')
  # disparity_map = cv2.medianBlur(disparity_map, parameters.bsize[0])
  # cv2.imwrite(output_name, disparity_map)

  # dusk = t.time()
  # print('\nFin.')
  # print('\nTotal execution time = {:.2f} s'.format(dusk - dawn))
