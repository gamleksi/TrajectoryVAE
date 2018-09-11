import torch
import torch.optim as optim
from trajectory_loader import TrajectoryLoader

from model import TrajectoryVAE
from monitor import Trainer
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for blender data')
parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent_size', default=4, type=int, help='Number of latent variables')
parser.add_argument('--num_epoch', default=2, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=552, type=int)
parser.add_argument('--num_workers', default=18, type=int)
parser.add_argument('--beta', default=4, type=int)
parser.add_argument('--num_actions', default=20, type=int)

#parser.add_argument('--gamma', default=300, type=int)
#parser.add_argument('--capacity_limit', default=30, type=int)
#parser.add_argument('--capacity_change_duration', default=60000, type=int)
#
#parser.add_argument('--capacity', dest='capacity', action='store_true')
#parser.add_argument('--no-capacity', dest='capacity', action='store_false')
#parser.set_defaults(capacity=False)

parser.add_argument('--folder_name', default='trajectory_vae', type=str)

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--visdom', dest='visdom', action='store_true')
parser.add_argument('--no-visdom', dest='visdom', action='store_false')
parser.set_defaults(visdom=True)

parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)

args = parser.parse_args()

LEARNING_RATE = args.lr
NUM_LATENT_VARIABLES = args.latent_size
folder_name = args.folder_name
beta = args.beta

#use_capacity = args.capacity
#gamma = args.gamma
#capacity_limit = args.capacity_limit
#capacity_change_duration = args.capacity_change_duration

#if use_capacity:
#    file_name = 'model_g_{}_lim_{}_dur_{}_l_{}_lr_{}'.format(gamma, capacity_limit, capacity_change_duration, NUM_LATENT_VARIABLES, LEARNING_RATE)
#else:
file_name = 'model_b_{}_l_{}_lr_{}'.format(beta, NUM_LATENT_VARIABLES, LEARNING_RATE)

NUM_EPOCHS = args.num_epoch
BATCH_SIZE = args.batch_size
NUM_PROCESSES = args.num_workers
debug = args.debug
visdom = args.visdom
log = args.log
num_actions = args.num_actions

def main():

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    model = TrajectoryVAE(NUM_LATENT_VARIABLES, num_actions, 7, device, beta=beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = TrajectoryLoader(BATCH_SIZE, NUM_PROCESSES, num_actions, debug=args.debug)

    trainer = Trainer(dataloader, model, NUM_LATENT_VARIABLES, save_folder=folder_name, save_name=file_name, log=log, visdom=visdom, visdom_title=folder_name, debug=debug)
    trainer.train(NUM_EPOCHS, optimizer)

if __name__ == '__main__':
    main()
