#!/usr/bin/python3

import copy
import numpy as np
import PIL
import pickle
import torch
import random

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from typing import Any, List, Tuple


from proj4_code.disparity_map import calculate_disparity_map
from proj4_code.similarity_measures import ssd_similarity_measure, sad_similarity_measure
"""
File adapted from project 2
"""


def generate_random_stereogram(im_size: Tuple[int, int, int] = (51, 51, 3), disparity: int = 4) -> torch.Tensor:
  """
  Generates a random stereogram for the given image size. The block which is 
  shifted is centered at the image center and of 0.5 times the dimension of the input.

  Steps:
  1. Generate the left image to be random 0/1 values
  2. Set the right image as the copy of the left image
  3. Move a block around the center block in the right image by 'disparity val' to the left
  4. Fill in the 'hole' in the right image with random values

  Note: 
  1. The block to be moved is a square of size (H//2,W//2) at the center pixel of the image (H,W,C)
     Note the use of integer division.
  2. The values in the images should be 0 and 1 (at random)
  3. Your code will not be tested with inputs where moving the block with the given disparity
     takes the block out of bounds.
  4. The resulting image should be grayscale, i.e. a pixel value should be same in all the channels.
     image[x,y,0] == image[x,y,1] == ..... and so on for all the channels

  Args:
  - im_size: The size of the image to be be generated
  - disparity: the shift to be induced in the right image
  Returns:
  - im_left: the left image as a torch tensor
  - im_right: the right image as a torch tensor
  """

  H, W, C = im_size
  block_size = (H//2, W//2)
  im_left = torch.zeros(1) #placeholder, not actual size
  im_right = torch.zeros(1) #placeholder, not actual size
  ############################################################################
  # Student code begin
  ############################################################################

  raise NotImplementedError("generate_random_stereogram not implemented")

  ############################################################################
  # Student code begin
  ############################################################################
  return im_left, im_right


def stereo_helper_fn(im_left, im_right, block_size=[5, 9, 13], max_search_bound=15):
  '''
  This helper function will help us in calculating disparity maps for different parameters.
  It also plots the image.

  Please tune the parameters and see the effect of them for different inputs.

  Args:
    - im_left: the left image
    - im_right: the right image
    - block_size: list of different block sizes to be used
    - max_search_bound: the max horizontal displacement to look for the most similar patch
                        (Refer to the project webpage for more details)
  '''
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

  ax1.imshow(im_left, interpolation=None)
  ax1.title.set_text('Left image')
  ax1.autoscale(False)
  ax1.set_axis_off()

  ax2.imshow(im_right, interpolation=None)
  ax2.title.set_text('Right image')
  ax2.autoscale(False)
  ax2.set_axis_off()

  plt.show()

  # fig, ax = plt.subplots(len(block_size),2, figsize=(15, 10*len(block_size)))

  for idx, block in enumerate(block_size):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
    # **calculate the disparity maps**

    # Using SAD similarity function
    disp_map_sad = calculate_disparity_map(im_left,
                                           im_right,
                                           block_size=block,
                                           sim_measure_function=sad_similarity_measure,
                                           max_search_bound=max_search_bound)

    # Using SSD similarity function
    disp_map_ssd = calculate_disparity_map(im_left,
                                           im_right,
                                           block_size=block,
                                           sim_measure_function=ssd_similarity_measure,
                                           max_search_bound=max_search_bound)

    im = ax1.imshow(disp_map_sad, cmap='jet')
    ax1.set_title('Disparity Map - SAD ({}x{} patch)'.format(block, block))
    ax1.autoscale(True)
    ax1.set_axis_off()
    cbar = fig.colorbar(im, ax=ax1, cmap='jet', shrink=0.3)

    im = ax2.imshow(disp_map_ssd, cmap='jet')
    ax2.set_title('Disparity Map - SSD ({}x{} patch)'.format(block, block))
    ax2.autoscale(True)
    ax2.set_axis_off()
    cbar = fig.colorbar(im, ax=ax2, cmap='jet', shrink=0.3)

    plt.show()


def rgb2gray(img: np.ndarray) -> np.ndarray:
  """ 
  Use the coefficients used in OpenCV, found here:
  https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

  Args:
  -   Numpy array of shape (M,N,3) representing RGB image

  Returns:
  -   Numpy array of shape (M,N) representing grayscale image
  """
  # Grayscale coefficients
  c = [0.299, 0.587, 0.114]
  return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]


def PIL_resize(img: np.ndarray, ratio: Tuple[float, float]) -> np.ndarray:
  """
  Args:
  - img: Array representing an image
  - size: Tuple representing new desired (width, height)

  Returns:
  - img
  """
  H, W, _ = img.shape
  img = numpy_arr_to_PIL_image(img, scale_to_255=True)
  img = img.resize((int(W*ratio[1]), int(H*ratio[0])), PIL.Image.LANCZOS)
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

  for scale in range(2, scales + 1):
    # add padding
    img_scales = np.hstack((img_scales,
                            np.ones((original_height, padding, num_colors),
                                    dtype=np.float32)))

    new_h = int(scale_factor * new_h)
    new_w = int(scale_factor * new_w)
    # downsample image iteratively
    cur_image = 2(cur_image, size=(new_w, new_h))

    # pad the top to append to the output
    h_pad = original_height - cur_image.shape[0]
    pad = np.ones((h_pad, cur_image.shape[1], num_colors),
                  dtype=np.float32)
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


def hstack_images(img1, img2):
  """
  Stacks 2 images side-by-side and creates one combined image.

  Args:
  - imgA: A numpy array of shape (M,N,3) representing rgb image
  - imgB: A numpy array of shape (D,E,3) representing rgb image

  Returns:
  - newImg: A numpy array of shape (max(M,D), N+E, 3)
  """

  # CHANGED
  imgA = np.array(img1)
  imgB = np.array(img2)
  Height = max(imgA.shape[0], imgB.shape[0])
  Width = imgA.shape[1] + imgB.shape[1]

  newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
  newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
  newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

  # newImg = PIL.Image.fromarray(np.uint8(newImg))
  return newImg


def generate_delta_fn_images(im_size):
  """
  Generates a pair of left and right (stereo pair) images of a single point.
  This point mimics a delta function and will manifest as a single pixel
  on the same vertical level in both the images. The horizontal distance
  between the pixels will be proportial to the 3D depth of the image
  """

  H = im_size[0]
  W = im_size[1]

  im1 = torch.zeros((H, W, 3))
  im2 = torch.zeros((H, W, 3))

  # pick a location of a pixel in im1 randomly
  im1_r = random.randint(0, H - 1)
  im1_c = random.randint(W // 2, W - W//4)

  im1[im1_r, im1_c, :] = torch.FloatTensor([1.0, 1.0, 1.0])

  # pick a location of the pixel in im2
  im2_r = im1_r
  im2_c = im1_c - random.randint(1, W // 4 - 1)

  im2[im2_r, im2_c, :] = torch.FloatTensor([1.0, 1.0, 1.0])

  return im1, im2
