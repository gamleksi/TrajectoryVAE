import os
import torch
import torch.nn as nn
import csv
import numpy as np
import torch.optim as optim
from trajectory_loader import TrajectoryLoader

from model import TrajectoryVAE
from monitor import Trainer
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for Trajectory generation')

parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent_size', default=4, type=int, help='Number of latent variables')
parser.add_argument('--num-epoch', default=20, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=552, type=int)
parser.add_argument('--num_workers', default=18, type=int)

parser.add_argument('--num-actions', default=24, type=int)
parser.add_argument('--num-joints', default=7, type=int)

parser.add_argument('--dataset-name', default='example', type=str)
parser.add_argument('--dataset-root', default='/home/aleksi/mujoco_ws/src/motion_planning/trajectory_data', type=str)

parser.add_argument('--folder-name', default='trajectory_test', type=str)

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)

parser.add_argument('--conv', dest='conv', action='store_true')
parser.add_argument('--no-conv', dest='conv', action='store_false')
parser.set_defaults(conv=False)

parser.add_argument('--conv-channel', default=2, type=int, help='1D conv out channel')
parser.add_argument('--kernel-row', default=4, type=int, help='Size of Kernel window in 1D')

parser.add_argument('--beta-interval', default=50, type=int, help='Step size for updating a beta value')
parser.add_argument('--beta-min', default=1.0e-4, type=float, help='Initial value of beta')
parser.add_argument('--beta-max', default=1.0, type=float, help='End value of beta')


def use_cuda():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')
    return torch.device('cuda' if use_cuda else 'cpu')


def save_arguments(args):
    save_path = os.path.join('log', args.folder_name)
    args = vars(args)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    w = csv.writer(open(os.path.join(save_path, "arguments.csv"), "w"))
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


def get_dataset_path(folder_name, dataset_root):
    return os.path.join(dataset_root, folder_name, 'trajectories.pkl')


def define_model_name(latent_size, lr):

    file_name = 'model_l_{}_lr_{}'.format(latent_size, lr)
    return file_name


np.random.seed(seed=7)
torch.manual_seed(7)

def main(args):

    # Model parameters
    lr = args.lr
    latent_size = args.latent_size
    num_actions = args.num_actions
    num_joints = args.num_joints

    # Training parameter
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    num_processes = args.num_workers
    debug = args.debug

    device = use_cuda()

    # Model Save
    log_folder = args.folder_name
    model_name = define_model_name(latent_size, lr)
    do_log = args.log

    if do_log:
        save_arguments(args)

    if debug:
        dataset_path = get_dataset_path('example', args.dataset_root)
    else:
        dataset_path = get_dataset_path(args.dataset_name, args.dataset_root)

    assert(os.path.exists(dataset_path))

    model = TrajectoryVAE(latent_size, num_actions, num_joints, device, num_epoch=num_epoch,
                          conv_model=args.conv, kernel_row=args.kernel_row,
                          conv_channel=args.conv_channel, beta_interval=args.beta_interval,
                          beta_min=args.beta_min, beta_max=args.beta_max).to(device)

    gauss_init(model.encoder)
    gauss_init(model.decoder)

    dataloader = TrajectoryLoader(batch_size, num_processes, dataset_path, actions_per_trajectory=num_actions)

    trainer = Trainer(dataloader, model, save_folder=log_folder, save_name=model_name, log=do_log, debug=debug)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer.train(num_epoch, optimizer)

if __name__ == '__main__':

    args = parser.parse_args()

    main(args)
