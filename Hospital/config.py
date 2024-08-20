import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Hospital Seizure Classification')

    parser.add_argument('--set', default="overlap", type=str,
                    help='Settings for overlap.')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N',
                        help='number of workers at dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    parser.add_argument('--scales', default=128, type=int, help='frequency scales range')
    parser.add_argument('--epochs', default = 200, type=int, help='number of epoch')



    parser.add_argument('--seed',default = '1234', type=int)
    parser.add_argument('--save_model', action='store_true', default=False)    

    return parser
