import os
import torch
import torch.nn as nn
import csv
import numpy as np
import torch.optim as optim
from trajectory_loader import TrajectoryLoader

from trajectory_vae import TrajectoryVAE
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for Trajectory generation')

parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent_size', default=4, type=int, help='Number of latent variables')
parser.add_argument('--num-epoch', default=20, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=552, type=int)
parser.add_argument('--num_workers', default=18, type=int)

parser.add_argument('--num-actions', default=24, type=int)
parser.add_argument('--num-joints', default=7, type=int)

parser.add_argument('--dataset-path', default='example', type=str, help='Path to trajectories.pkl')

parser.add_argument('--model-name', default='test_v1', type=str, help='Folder name of the model result')
parser.add_argument('--log-path', default='traj_vae_results', type=str, help='Root path to results')

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)


'''
1-D convolution structure did not improve the learning performance  

parser.add_argument('--conv', dest='conv', action='store_true')
parser.add_argument('--no-conv', dest='conv', action='store_false')
parser.set_defaults(conv=False)

parser.add_argument('--conv-channel', default=2, type=int, help='1D conv out channel')
parser.add_argument('--kernel-row', default=4, type=int, help='Size of Kernel window in 1D')
'''


parser.add_argument('--beta-interval', default=50, type=int, help='Step size for updating a beta value')
parser.add_argument('--beta-min', default=1.0e-8, type=float, help='Initial value of beta')
parser.add_argument('--beta-max', default=1.0e-5, type=float, help='End value of beta')


def use_cuda():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')
    return torch.device('cuda' if use_cuda else 'cpu')


def save_arguments(args, model_path):
    args = vars(args)
    if not(os.path.exists(model_path)):
        os.makedirs(model_path)
    w = csv.writer(open(os.path.join(model_path, "arguments.csv"), "w"))
    for key, val in args.items():
        w.writerow([key, val])


def gauss_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def get_dataset_path(dataset_path):
    return os.path.join(dataset_path, 'trajectories.pkl')


np.random.seed(seed=7)
torch.manual_seed(7)

def main(args):

    device = use_cuda()

    model_path = os.path.join(args.log_path, args.model_name)

    # Model Save
    if args.log:
        save_arguments(args, model_path)

    dataset_path = get_dataset_path(args.dataset_path)

    assert(os.path.exists(dataset_path))

    model = TrajectoryVAE(args.latent_size, args.num_actions, args.num_joints, device,
                          num_epoch=args.num_epoch, beta_interval=args.beta_interval, beta_min=args.beta_min, beta_max=args.beta_max).to(device)

    gauss_init(model.encoder)
    gauss_init(model.decoder)

    dataloader = TrajectoryLoader(args.batch_size, args.num_workers, dataset_path, actions_per_trajectory=args.num_actions)

    trainer = Trainer(dataloader, model, model_path, log=args.log, debug=args.debug)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer.train(args.num_epoch, optimizer)


if __name__ == '__main__':

    args = parser.parse_args()

    main(args)
