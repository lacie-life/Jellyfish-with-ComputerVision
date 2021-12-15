"""
Tests for similarity measures
"""

import torch

from proj4_code.similarity_measures import ssd_similarity_measure, sad_similarity_measure


def test_ssd_similarity_measure_values():
  """
  Unit tests for SSD similarity measure using precomputed values
  """

  patch1 = torch.tensor([1.3, 4.5, 7.2, 0.2, -0.6])
  patch2 = torch.tensor([0.2, 4.4, 7.6, 0.1, 1.3])

  ssd = ssd_similarity_measure(patch1, patch2)
  assert abs(ssd.item()-5.0) <= 1e-5


def test_sad_similarity_measure_values():
  """
  Unit tests for SSD similarity measure using precomputed values
  """

  patch1 = torch.tensor([1.3, 4.5, 7.2, 0.2, -0.6])
  patch2 = torch.tensor([0.2, 4.4, 7.6, 0.1, 1.3])

  sad = sad_similarity_measure(patch1, patch2)

  assert abs(sad.item()-3.6) <= 1e-5


def test_similarity_measure_size_compatibility():
  """
  Unit tests for SSD and SAD similarity functions flexibility with different sizes
  """

  patch1 = torch.randn(size=(4, 6, 2))
  patch2 = torch.randn(size=(4, 6, 2))

  ssd_similarity_measure(patch1, patch2)
  sad_similarity_measure(patch1, patch2)
  assert True  # just check if the ssd calculation was successfull

  patch1 = torch.randn(size=(4, 3))
  patch2 = torch.randn(size=(4, 3))

  ssd_similarity_measure(patch1, patch2)
  sad_similarity_measure(patch1, patch2)
  assert True  # just check if the ssd calculation was successfull

  patch1 = torch.randn(size=(5,))
  patch2 = torch.randn(size=(5,))

  ssd_similarity_measure(patch1, patch2)
  sad_similarity_measure(patch1, patch2)
  assert True  # just check if the ssd calculation was successfull

  patch1 = torch.randn(size=(3, 7, 2, 4))
  patch2 = torch.randn(size=(3, 7, 2, 4))

  ssd_similarity_measure(patch1, patch2)
  sad_similarity_measure(patch1, patch2)
  assert True  # just check if the ssd calculation was successfull
