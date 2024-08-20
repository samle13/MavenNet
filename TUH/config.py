import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description='TUH Seizure Classification')

    parser.add_argument('--num_workers', default=12, type=int, metavar='N',
                        help='number of workers at dataloader')
    
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    parser.add_argument('--epochs', default = 50, type=int, help='number of epoch')



    return parser
