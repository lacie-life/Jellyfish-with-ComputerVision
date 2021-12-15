'''
Utilities to be used along with the deep model
'''

import torch


def predict_labels(model: torch.nn.Module, x: torch.tensor) -> torch.tensor:
  '''
  Perform the forward pass and extract the labels from the model output

  Args:
  -   model: a model (which inherits from nn.Module)
  -   x: the input image [Dim: (N,C,H,W)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''

  predicted_labels = None

  #############################################################################
  # Student code begin
  #############################################################################

  raise NotImplementedError('predict_labels not implemented')

  #############################################################################
  # Student code end
  #############################################################################
  return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
  '''
  Computes the loss between the model output and the target labels

  Args:
  -   model: a model (which inherits from nn.Module)
  -   model_output: the raw scores output by the net
  -   target_labels: the ground truth class labels
  -   is_normalize: bool flag indicating that loss should be divided by the batch size
  Returns:
  -   the loss value
  '''

  loss = None

  #############################################################################
  # Student code begin
  #############################################################################

  raise NotImplementedError('compute_loss not implemented')

  #############################################################################
  # Student code end
  #############################################################################
  return loss
