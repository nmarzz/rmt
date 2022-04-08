""" Holds model classes and a getter function to access them """

from torch import nn
from argparse import Namespace
import torch.nn.functional as F
import torchvision
import torch
import numpy as np

from dropout import RBFDrop, BernDrop, SineDrop

class MLP(nn.Module):
    ''' A basic 3 layer MLP '''

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout_proportion : float = None, dropout_type : str = 'k_bernoulli') -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim * 4)        
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.register_buffer('rescaling', torch.zeros(hidden_dim))
        self.batches_trained_for = 0
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout_type = dropout_type
        self.dropout_proportion = dropout_proportion
        self.num_layers = 3

        # Keep this around to compare to torch's dropout
        if self.dropout_type == 'pytorch':
            self.d1 = nn.Dropout(p=self.dropout_proportion)            


        ### Build in dropout classes
        if self.dropout_type == 'rbf':
            self.dropout = RBFDrop(self.dropout_proportion)
        elif self.dropout_type == 'k_bernoulli':
            self.dropout = BernDrop(self.dropout_proportion)
        elif self.dropout_type == 'sine':
            self.dropout = SineDrop(self.dropout_proportion)
    

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))                               
        x = F.relu(self.fc2(x))        

        if self.training:            
            if self.dropout_type == 'pytorch':
                x = self.d1(x)
            elif self.dropout_type != 'no_dropout':                            
                x, dropped_indices = self.dropout.apply_dropout(x)
                self.batches_trained_for += 1
                self.rescaling[dropped_indices] += 1
                # print(self.rescaling)
                # print(self.batches_trained_for)
                # print( self.rescaling / self.batches_trained_for)
        else:
            if (self.dropout_type == 'pytorch') or (self.dropout_type == 'no_dropout'):
                pass
            elif self.dropout_type != 'no_dropout':                            
                # Rescale the output to account for the dropout process                    
                x = x * self.rescaling / self.batches_trained_for         

        x = self.fc3(x)

        return x

    def output_embeds(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))

        x1 = x.clone()        
        x = F.relu(self.fc2(x))                
        x2 = x.clone()
                                
        if self.training:            
            if self.dropout_type == 'pytorch':
                x = self.d1(x)
            elif self.dropout_type != 'no_dropout':                            
                x, dropped_indices = self.dropout.apply_dropout(x)
                self.batches_trained_for += 1
                self.rescaling[dropped_indices] += 1
        else:
            if (self.dropout_type == 'pytorch') or (self.dropout_type == 'no_dropout'):
                pass
            elif self.dropout_type != 'no_dropout':                            
                # Rescale the output to account for the dropout process                    
                x = x * self.rescaling / self.batches_trained_for 

        x3 = x.clone()
        return x1,x2,x3
    
    @property
    def device(self):
        return next(self.parameters()).device


def get_embeddings(model: nn.Module, loader):
    """Get layer's embeddings on data in numpy format """                
    model.eval()
    device = model.device
    
    embeddings = [None for _ in range(model.num_layers)]    

    with torch.no_grad():
        for data, _ in loader:            
            data = data.to(device)
            embs = model.output_embeds(data)
            for i,e in enumerate(embs):
                if embeddings[i] is None:
                    embeddings[i] = e.cpu().numpy()
                else:
                    embeddings[i] = np.concatenate([embeddings[i], e.cpu().numpy()], axis = 0)
       
    return embeddings


def get_model(model_type: str, args: Namespace):
    ''' Getter function to return the appropraite version of a model '''

    if (args.dataset == 'mnist') or (args.dataset == 'fashion_mnist'):
        input_dim = 28*28
        num_channels = 1
        num_classes = 10
    elif (args.dataset == 'cifar10'):
        input_dim = 3*32*32
        num_channels = 3
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
