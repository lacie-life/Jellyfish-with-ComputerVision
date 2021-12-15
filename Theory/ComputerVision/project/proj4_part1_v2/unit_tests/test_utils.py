"""
Tests for the utils module
"""
import torch

from proj4_code.utils import generate_random_stereogram


def test_generate_random_stereogram():
  """
  Tests the generation of random dot stereogram
  """
  H, W = (51, 51)
  disparity_val = 4
  left_img, right_img = generate_random_stereogram(
      im_size=(H, W, 2), disparity=disparity_val)

  # assert that they are same in all the channels
  for ch_idx in range(1, left_img.shape[2]):
    assert torch.nonzero(left_img[:, :, 0] !=
                         left_img[:, :, ch_idx]).shape[0] == 0
    assert torch.nonzero(right_img[:, :, 0] !=
                         right_img[:, :, ch_idx]).shape[0] == 0

  diff_img = torch.abs(left_img - right_img)

  # get the region where left and right images are different
  nonzero_idx = torch.nonzero(diff_img)
  falsevals = torch.nonzero(~(
      (nonzero_idx[:, 0] >= 13) &
      (nonzero_idx[:, 0] <= 37) &
      (nonzero_idx[:, 1] >= 9) &
      (nonzero_idx[:, 1] <= 37)
  ))

  assert falsevals.shape[0] == 0
