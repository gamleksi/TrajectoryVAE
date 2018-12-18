import os

import numpy as np
from ros_monitor import ROSTrajectoryVAE
from torch.autograd import Variable
from trajectory_loader import TrajectoryLoader
import matplotlib.pyplot as plt
import argparse

def get_dataset_path(folder_name, dataset_root):
    return os.path.join(dataset_root, folder_name, 'trajectories.pkl')

def main(args):
    dataset_path = get_dataset_path('lumi', 'dset')

    model = ROSTrajectoryVAE(args.model_name, args.latent_size, args.num_actions, num_joints=args.num_joints)
    vae = model.model
    loader = TrajectoryLoader(785, 16, dataset_path, actions_per_trajectory=args.num_actions)
    iter = loader.get_iterator(True)

    latents = []

    for b in iter:
        trajectory = vae.to_torch(b)
        x = Variable(trajectory).to(vae.device)
        recon,  latent, _ = vae._forward(x, False)
        latent = latent.cpu().detach().numpy()
        latents.append(latent.transpose(1, 0))

    latents = np.concatenate([l for l in latents], axis=1)
    fig, axes = plt.subplots(args.latent_size, 1, sharex=True, figsize=[30, 30])

    for i in range(args.latent_size):
        ax = axes[i]
        batch = latents[i]
        ax.hist(batch, 600)
        ax.set_title('Latent {}'.format(i + 1))
        ax.set_xlabel('x')
        ax.set_ylabel('frequency')

    fig.tight_layout(pad=2)
    plt.savefig(os.path.join(os.path.join('log', args.model_name, 'latent_distribution.png')))
    plt.close()


parser = argparse.ArgumentParser(description='Model debugger: latents distribution')
parser.add_argument('--latent-size', default=4, type=int, help='Number of latent variables')
parser.add_argument('--num-joints', default=7, type=int)
parser.add_argument('--num-actions', default=24, type=int)
parser.add_argument('--model-name', default="mse_fc_v1", type=str)

if __name__ == '__main__':
    main(parser.parse_args())
