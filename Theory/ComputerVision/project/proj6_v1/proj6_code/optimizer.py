'''
This class contains helper functions which will help get the optimizer
'''

import torch


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
  '''
  Returns the optimizer initializer according to the config

  Note: config has a minimum of three entries.
  Feel free to add more entries if you want.
  But do not change the name of the three existing entries

  Args:
  - model: the model to optimize for
  - config: a dictionary containing parameters for the config
  Returns:
  - optimizer: the optimizer
  '''

  optimizer = None

  optimizer_type = config.get("optimizer_type", "sgd")
  learning_rate = config.get("lr", 1e-20)
  weight_decay = config.get("weight_decay", 1e3)

  ############################################################################
  # Student code begin
  ############################################################################

  raise NotImplementedError('get_optimizer not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return optimizer
