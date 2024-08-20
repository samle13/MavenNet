import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description='CHB-MIT Seizure Detection')

    parser.add_argument('--set', default="overlap", type=str,
                    help='Settings for overlap.')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N',
                        help='number of workers at dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    parser.add_argument('--scales', default=64, type=int, help='frequency scales range')
    parser.add_argument('--epochs', default = 200, type=int, help='number of epoch')

    parser.add_argument('--seed',default = '1234', type=int)    

    return parser
