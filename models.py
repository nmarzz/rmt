""" Holds model classes and a getter function to access them """

from torch import nn
from argparse import Namespace
import torch.nn.functional as F
import torchvision
import torch

from dropout import dropout

class MLP(nn.Module):
    ''' A basic 3 layer MLP '''

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout_proportion : int = None, dropout_type : str = 'k_bernoulli') -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout_type = dropout_type
        self.dropout_proportion = dropout_proportion

        # Keep this around to compare to torch's dropout
        if self.dropout_type == 'pytorch':
            self.d1 = nn.Dropout(p=self.dropout_proportion)
            self.d2 = nn.Dropout(p=self.dropout_proportion)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))        
                
        if self.training:            
            if self.dropout_type == 'pytorch':
                x = self.d1(x)
            else:
                x = dropout(x,x,self.dropout_proportion, self.dropout_type)

        x = F.relu(self.fc2(x))                
        if self.training:            
            if self.dropout_type == 'pytorch':
                x = self.d1(x)
            else:
                x = dropout(x,x,self.dropout_proportion, self.dropout_type)
        
        x = self.fc3(x)
        return x


def get_model(model_type: str, args: Namespace):
    ''' Getter function to return the appropraite version of a model '''

    if (args.dataset == 'mnist') or (args.dataset == 'fashion_mnist'):
        input_dim = 28*28
        num_channels = 1
        num_classes = 10
    else:
        raise ValueError(
            f'Model hyper-parameters unavailable for dataset {args.dataset}')

    if model_type == 'mlp':
        model = MLP(input_dim, num_classes, dropout_proportion = args.dropout_proportion, dropout_type = args.dropout_type)    
    else:
        return ValueError(f'Model {args.model} is unavailable')

    if (args.load_path is not None):
        model.load_state_dict(torch.load(args.load_path))

    return model
