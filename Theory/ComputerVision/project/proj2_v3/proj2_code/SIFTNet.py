#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from torch import nn
from proj2_code.torch_layer_utils import ImageGradientsLayer


"""
Authors: John Lambert, Vijay Upadhya, Patsorn Sangkloy, Cusuh Ham,
Frank Dellaert, September 2019.

Implement the SIFT Deep Net that accomplishes the identical operations as the
original SIFT algorithm (See Szeliski 4.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf


Your implementation does not need to exactly match the SIFT reference. For
example, we will be excluding scale and rotation invariance. However, here are
the key properties your descriptor should have:

(1) a 4x4 grid of cells, each feature_width/4. It is simply the
    terminology used in the feature literature to describe the spatial
    bins where gradient distributions will be described.
(2) each cell should have a histogram of the local distribution of
    gradients in 8 orientations. Appending these histograms together will
    give you 4x4 x 8 = 128 dimensions.
(3) Each feature should be normalized to unit length.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells
As described in Szeliski, a single gradient measurement creates a
weighted contribution to the 4 nearest cells and the 2 nearest
orientation bins within each cell, for 8 total contributions. This type
of interpolation probably will help, though.

Instead of explicitly computing the gradient orientation at each
pixel, we wish for you to instead filter with oriented filters (e.g. a
filter that responds to edges with a specific orientation). All of your
SIFT-like feature can be constructed entirely from filtering fairly quickly in
this way.

Regarding subgrid size -- a 4x4 filter is undesirable since it has even
dimensions, necessitating asymmetric padding along a single axis.

However, the impact is negligible -- the performance with a 5x5 filter is
identical, so we stick with the original 4x4 implementation.

You can find a review of what a conv layer does here:
    http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture05.pdf
"""


class HistogramLayer(nn.Module):
    def __init__(self) -> None:
        """
        Initialize parameter-less histogram layer, that accomplishes
        per-channel binning.

        Args:
        -   None

        Returns:
        -   None
        """
        super(HistogramLayer, self).__init__()

    def forward(self, x) -> torch.Tensor:
        """
        Complete a feedforward pass of the histogram/binning layer byforming a
        weighted histogram at every pixel value.

        The input should have 10 channels, where the first 8 represent cosines
        values of angles between unit circle basis vectors and image gradient
        vectors, at every pixel. The last two channels will represent the
        (dx, dy) coordinates of the image gradient at this pixel.

        The weighted histogram can be created by elementwise multiplication of
        a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
        tensor, where a tensor cell is activated if its value represents the
        maximum channel value within a "fibre" (see
        http://cs231n.github.io/convolutional-networks/ for an explanation of a
        "fibre"). There will be a fibre (consisting of all channels) at each of
        the (M,N) pixels of the "feature map".

        The four dimensions represent (N,C,H,W) for batch dim, channel dim,
        height dim, and weight dim, respectively. Our batch size will be 1.

        In order to create the 4d binary occupancy tensor, you may wish to
        index in at many values simultaneously in the 4d tensor, and read or
        write to each of them simultaneously. This can be done by passing a 1d
        Pytorch Tensor for every dimension, e.g. by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

        You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
        flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
        torch.norm() helpful.

        With a double for-loop you could expect 20 sec. runtime for this
        function. You may not submit code with a triple for-loop (which would
        take over 60 seconds). With tensor indexing, this should take 0.08-0.11
        sec.

        ** You will receive extra-credit if you successfully implement this
        function with no for-loops (fully-vectorized code). However, if you
        can't get it the vectorized version to work, please submit the working
        version with two for-loops.

        Args:
        -   x: tensor with shape (1,10,M,N), where M,N are height, width

        Returns:
        -   per_px_histogram: tensor with shape (1,8,M,N) representing a weighted
            histogram at every pixel
        """
        cosines = x[:,:8,:,:] # Contains
        im_grads = x[:,8:,:,:] # Contains dx, dy

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        raise NotImplementedError('`HistogramLayer.forward` function in '
            + '`student_sift.py` needs to be implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return per_px_histogram


class SubGridAccumulationLayer(nn.Module):
    """
    """
    def __init__(self) -> None:
        """
        Given 8-dimensional feature vectors at each pixel, accumulate features
        over 4z4 subgrids.

        You may find the Pytorch function nn.Conv2d() helpful here. In Pytorch,
        a Conv2d layer's behavior is governed by the `groups` parameter. You
        will definitely need to understand the effect of this parameter. With
        groups=1, if your input is 28x28x8, and you wish to apply a 5x5 filter,
        then you will be convolving all inputs to all outputs (i.e. you will be
        convolving a 5x5x8 filter at every possible location over the feature
        map. However, if groups=8, then you will be convolving a 5x5x1 filter
        over each channel separately.

        Args:
        -   None

        Returns:
        -   None
        """
        super(SubGridAccumulationLayer, self).__init__()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        raise NotImplementedError('`__init__ in `SubGridAccumulationLayer` '
          + 'needs to be implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the SubGridAccumulationLayer().

        Args:
        -   x: Torch tensor representing an (b,c,m,n) layer, where b=1, c=8

        Returns:
        -   out: Torch tensor representing an (b,c,m,n) layer, where b=1, c=8
        """

        return self.layer(x)


def angles_to_vectors_2d_pytorch(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to 2-d basis vectors.
    You may find torch.cat(), torch.cos(), torch.sin() helpful.

    Args:
    -   angles: Torch tensor of shape (N,) representing N angles, measured in
        radians

    Returns:
    -   angle_vectors: Torch tensor of shape (N,2), representing x- and y-
        components of unit vectors in each of N angles, provided as argument.
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`angles_to_vectors_2d_pytorch` needs to be '
      + 'implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return angle_vectors


class SIFTOrientationLayer(nn.Module):
    """
    SIFT analyzes image gradients according to 8 bins, around the unit circle
    (a polar grid).
    """
    def __init__(self):
        """
        Initialize the model's layers and populate the layer weights
        appropriately. You should have 10 filters in the batch dimension.

        You may find the Pytorch function nn.Conv2d() helpful here.

        Args:
        -   None

        Returns:
        -   None
        """
        super(SIFTOrientationLayer, self).__init__()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        raise NotImplementedError('`__init__` in `SIFTOrientationLayer` needs '
          + 'to be implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def get_orientation_bin_weights(self) -> torch.nn.Parameter():
        """
        Populate the conv layer weights for the

        A 1x1 convolution layer makes perfect sense. For example, consider a
        1x1 CONV with 32 filters. Suppose your input is (1,64,56,56) in NCHW
        order. Then each filter has size (64,1,1) and performs a 64-dimensional
        dot product, producing a (1,32,56,56) tensor. In other words, you are
        performing a dot-product of two vectors with dim-64, and you do this
        with 32 different bases. This can be thought of as a 32x64 weight
        matrix.

        Args:
        -   None

        Returns:
        -   weight_param: Torch nn.Parameter, containing (10,2) matrix for the
            1x1 convolution's dot product
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        raise NotImplementedError('`get_orientation_bin_weights` needs to be '
          + 'implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return weight_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the SIFTOrientationLayer().

        Args:
        -   x: Torch tensor with shape (1,2,m,n)

        Returns:
        -   out: Torch tensor with shape (1,10,m,n)
        """
        return self.layer(x)


class SIFTNet(nn.Module):

    def __init__(self):
        """
        See http://cs231n.github.io/convolutional-networks/ for more details on
        what a conv layer does.

        Create a nn.Sequential() network, using the 4 specific layers you have
        implemented above. The layers above are not in any particular order.

        Args:
        -   None

        Returns:
        -   None
        """
        super(SIFTNet, self).__init__()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        raise NotImplementedError('`__init__` in `SIFTNet` needs to be '
          + 'implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SIFTNet. Feed a grayscale image through the SIFT
        network to obtain accumulated gradient histograms at every single
        pixel.

        Args:
        -   x: Torch tensor of shape (1,1,M,N) representing single grayscale
            image.

        Returns:
        -   Torch tensor representing 8-bin weighted histograms, accumulated
            over 4x4 grid cells
        """
        return self.net(x)


def get_sift_subgrid_coords(x_center: int, y_center: int):
    """
    Given the center point of a 16x16 patch, we eventually want to pull out the
    accumulated values for each of the 16 subgrids. We need the coordinates to
    do so, so return the 16 x- and y-coordinates, one for each 4x4 subgrid.

    Args:
    -   x_center: integer representing x-coordinate of keypoint.
    -   y_center: integer representing y-coordinate of keypoint.

    Returns:
    -   x_grid: (16,) representing x-coordinates
    -   y_grid: (16,) representing y-coordinates.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`get_sift_subgrid_coords` needs to be '
      + 'implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x_grid, y_grid


def get_siftnet_features(img_bw: torch.Tensor, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Given a list of (x,y) coordinates, pull out the SIFT features within the
    16x16 neighborhood around each (x,y) coordinate pair.

    Then normalize each 128-dimensional vector to have unit length.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one. Please raise each
    feature vector to the 0.9 power after normalizing.

    Args:
    -   img_bw: Torch tensor with shape (1,1,M,N) representing grayscale image.
    -   x: Numpy array with shape (K,)representing x-coordinates
    -   y: Numpy array with shape (K,)representing y-coordinates

    Returns:
    -   fvs: feature vectors of shape (K,128)
    """
    assert img_bw.shape[0] == 1
    assert img_bw.shape[1] == 1
    assert img_bw.dtype == torch.float32

    net = SIFTNet()
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`get_siftnet_features` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
