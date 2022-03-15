import torch.nn as nn


def get_loss_function(name: str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Loss function {name} not available')
