import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Bonn Seizure Detection')

    parser.add_argument('--set', default="ABvsCDvsE", type=str,
                    help='Directory where data files are stored.')
    parser.add_argument('--data_dir', default="./ABvsCDvsE", type=str,
                        help='Directory where data files are stored.')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='number of workers at dataloader')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--scales', default= 64, type=int)

    parser.add_argument('--epochs', default = 100, type=int, help='number of epoch')

    parser.add_argument('--seed',default = '1234', type=int)    
    parser.add_argument('--overlap',default = 'overlap', type=str)    


    return parser
