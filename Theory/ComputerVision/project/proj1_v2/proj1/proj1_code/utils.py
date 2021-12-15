#!/usr/bin/python3

import copy
import numpy as np
import PIL
import torch
import torchvision

from typing import Any, List, Tuple


def PIL_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
  """
    Args:
    - img: Array representing an image
    - size: Tuple representing new desired (width, height)

    Returns:
    - img
  """
  img = numpy_arr_to_PIL_image(img, scale_to_255=True)
  img = img.resize(size)
  img = PIL_image_to_numpy_arr(img)
  return img


def PIL_image_to_numpy_arr(img, downscale_by_255=True):
  """
    Args:
    - img
    - downscale_by_255

    Returns:
    - img
  """
  img = np.asarray(img)
  img = img.astype(np.float32)
  if downscale_by_255:
    img /= 255
  return img


def vis_image_scales_numpy(image: np.ndarray) -> np.ndarray:
  """
    This function will display an image at different scales (zoom factors). The
    original image will appear at the far left, and then the image will
    iteratively be shrunk by 2x in each image to the right.

    This is a particular effective way to simulate the perspective effect, as
    if viewing an image at different distances. We thus use it to visualize
    hybrid images, which represent a combination of two images, as described
    in the SIGGRAPH 2006 paper "Hybrid Images" by Oliva, Torralba, Schyns.

    Args:
    - image: Array of shape (H, W, C)

    Returns:
    - img_scales: Array of shape (M, K, C) representing horizontally stacked
      images, growing smaller from left to right.
      K = W + int(1/2 W + 1/4 W + 1/8 W + 1/16 W) + (5 * 4)
  """
  original_height = image.shape[0]
  original_width = image.shape[1]
  num_colors = 1 if image.ndim == 2 else 3
  img_scales = np.copy(image)
  cur_image = np.copy(image)

  scales = 5
  scale_factor = 0.5
  padding = 5

  new_h = original_height
  new_w = original_width

  for scale in range(2, scales+1):
    # add padding
    img_scales = np.hstack((img_scales,
      np.ones((original_height, padding, num_colors), dtype=np.float32))
    )

    new_h = int(scale_factor*new_h)
    new_w = int(scale_factor*new_w)
    # downsample image iteratively
    cur_image = PIL_resize(cur_image, size=(new_w, new_h))

    # pad the top to append to the output
    h_pad = original_height-cur_image.shape[0]
    pad = np.ones((h_pad, cur_image.shape[1], num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    img_scales = np.hstack((img_scales, tmp))

  return img_scales


def im2single(im: np.ndarray) -> np.ndarray:
  """
    Args:
    - img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
    - im: float or double array of identical shape and in range [0,1]
  """
  im = im.astype(np.float32) / 255
  return im

def single2im(im: np.ndarray) -> np.ndarray:
  """
    Args:
    - im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
    - im: uint8 array of identical shape and in range [0,255]
  """
  im *= 255
  im = im.astype(np.uint8)
  return im


def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
  """
    Args:
    - img: in [0,1]

    Returns:
    - img in [0,255]

  """
  if scale_to_255:
    img *= 255
  return PIL.Image.fromarray(np.uint8(img))



def load_image(path: str) -> np.ndarray:
  """
    Args:
    - path: string representing a file path to an image

    Returns:
    - float or double array of shape (m,n,c) or (m,n) and in range [0,1],
      representing an RGB image
  """
  img = PIL.Image.open(path)
  img = np.asarray(img)
  float_img_rgb = im2single(img)
  return float_img_rgb


def save_image(path: str, im: np.ndarray) -> bool:
  """
    Args:
    - path: string representing a file path to an image
    - img: numpy array

    Returns:
    - retval indicating write success
  """
  img = copy.deepcopy(im)
  img = single2im(img)
  pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
  return pil_img.save(path)


def write_objects_to_file(fpath: str, obj_list: List[Any]):
  """
    If the list contents are float or int, convert them to strings.
    Separate with carriage return.

    Args:
    - fpath: string representing path to a file
    - obj_list: List of strings, floats, or integers to be written out to a file, one per line.

    Returns:
    - None
  """
  obj_list = [str(obj) + '\n' for obj in obj_list]
  with open(fpath, 'w') as f:
    f.writelines(obj_list)
