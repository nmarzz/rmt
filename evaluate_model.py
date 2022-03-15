''' Utility to load and evaluate a model on a particular dataset.

Can handle:
    - decomposed models (with the --decomposed flag)
    - vanilla saved models
'''
import argparse
import torch
import numpy as np
from loaders import get_loader
from models import get_model
from training import predict
from loss_functions import get_loss_function


def get_args(parser):
    parser.add_argument('--split', type=str, choices=[
                        'train', 'val', 'test'], metavar='D', help='Choice of train, val, test')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--loss-function', type=str, default='cross_entropy')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--load-path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=1331, metavar='S')

    args = parser.parse_args()

    return args


def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Evaluate a model')
    args = get_args(parser)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    train_loader, val_loader = get_loader(
        args.dataset, args.batch_size, distributed=False)
    if args.split == 'train':
        loader = train_loader
    else:
        loader = val_loader

    model = get_model(args.model, args)

    model.to(device)

    metrics = predict(model, device, loader,
                      get_loss_function(args.loss_function))

    print('Loss: {}'.format(metrics[0]))
    print('Top-1 Accuracy: {}'.format(metrics[1]))
    print('Top-5 Accuracy: {}'.format(metrics[2]))


if __name__ == '__main__':
    main()
