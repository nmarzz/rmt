''' Main access point to training models in this project. '''

from training import GeneralTrainer
from models import get_model
from loaders import get_loader
from logger import Logger

import numpy as np
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp


def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--loss-function', type=str, default='cross_entropy')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--load-path', type=str)
    parser.add_argument('--pretrained', action='store_true')    
    parser.add_argument('--device', type=str, nargs='+', default=['cpu'],
                        help='Name of CUDA device(s) being used (if any). Otherwise will use CPU. \
                            Can also specify multiple devices (separated by spaces) for multiprocessing.')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', metavar='O')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)  
    parser.add_argument('--dropout-proportion', type=float, default=0.5)
    parser.add_argument('--dropout-type', type=str, default='k_bernoulli')        
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--early-stop', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    return args


def main_worker(idx: int, num_gpus: int, distributed: bool, args: argparse.Namespace):

    device = torch.device(args.device[idx])

    logger = Logger(args, save=(idx == 0))

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29501',
                                world_size=num_gpus, rank=idx)

    # Get the data
    batch_size = int(args.batch_size / num_gpus)
    train_loader, val_loader = get_loader(
        name=args.dataset, batch_size=batch_size, distributed=distributed, resize = args.pretrained)

    # Get model
    model = get_model(args.model, args)        
    model.to(device)

    trainer = GeneralTrainer(model=model, train_loader=train_loader,
                             val_loader=val_loader, device=device, logger=logger, idx=idx, args=args)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description='')
    args = get_args(parser)

    # # Set random seed for reproducibility
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # Launch training    
    main_worker(0, 1, False, args)


if __name__ == '__main__':
    main()
