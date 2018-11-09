import os
import torch
import torch.optim as optim
from trajectory_loader import TrajectoryLoader

from model import TrajectoryVAE
from monitor import Trainer
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for Trajectory generation')

parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent_size', default=4, type=int, help='Number of latent variables')
parser.add_argument('--num_epoch', default=20, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=552, type=int)
parser.add_argument('--num_workers', default=18, type=int)
parser.add_argument('--beta', default=4, type=float)

parser.add_argument('--num_actions', default=20, type=int)
parser.add_argument('--num_joints', default=7, type=int)

parser.add_argument('--dataset-name', default='example', type=str)
parser.add_argument('--dataset-root', default='/home/aleksi/mujoco_ws/src/motion_planning/trajectory_data', type=str)

parser.add_argument('--folder-name', default='trajectory_test', type=str)

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--visdom', dest='visdom', action='store_true')
parser.add_argument('--no-visdom', dest='visdom', action='store_false')
parser.set_defaults(visdom=True)

parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)

parser.add_argument('--simple', dest='simple', action='store_true')
parser.add_argument('--no-simple', dest='simple', action='store_false')
parser.set_defaults(debug=False)

def use_cuda():

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    return torch.device('cuda' if use_cuda else 'cpu')


def get_dataset_path(folder_name, dataset_root):
    return os.path.join(dataset_root, folder_name, 'trajectories.pkl')


def define_model_name(beta, latent_size, lr):
    file_name = 'model_b_{}_l_{}_lr_{}'.format(beta, latent_size, lr)
    return file_name

def main(args):

    # Model parameters
    lr = args.lr
    latent_size = args.latent_size
    beta = args.beta
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
    model_name = define_model_name(beta, latent_size, lr)
    visdom = args.visdom
    do_log = args.log

    if debug:
        dataset_path = get_dataset_path('example', args.dataset_root)
    else:
        dataset_path = get_dataset_path(args.dataset_name, args.dataset_root)

    assert(os.path.exists(dataset_path))

    model = TrajectoryVAE(latent_size, num_actions, num_joints, device, simple_model=args.simple, beta=beta).to(device)

    dataloader = TrajectoryLoader(batch_size, num_processes, dataset_path, actions_per_trajectory=num_actions)

    trainer = Trainer(dataloader, model, save_folder=log_folder, save_name=model_name, log=do_log,
                      visdom=visdom, visdom_title=log_folder)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer.train(num_epoch, optimizer)

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
