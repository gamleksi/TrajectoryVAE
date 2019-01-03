import os

import numpy as np
from ros_monitor import ROSTrajectoryVAE
from torch.autograd import Variable
from trajectory_loader import TrajectoryLoader
import matplotlib.pyplot as plt
import argparse


def get_dataset_path(folder_name, dataset_root):
    return os.path.join(dataset_root, folder_name, 'trajectories.pkl')


def trajectory_distributions(targets, reconstructions, file_name, args):

    assert(targets.shape[0] == reconstructions.shape[0])

    fig, axes = plt.subplots(args.num_joints, 2, sharex=True, sharey=True, figsize=[30, 30])
    steps = range(1, targets[0].shape[1] + 1)

    labels = ("targets", "reconstructed")

    for idx, trajectories in enumerate((targets, reconstructions)):
        for joint_idx in range(args.num_joints):
            ax = axes[joint_idx][idx]
            for traj_idx in range(len(targets)):
                trajectory = trajectories[traj_idx]
                ax.plot(steps, trajectory[joint_idx])
            ax.set_title("{} Joint {}".format(labels[idx], idx + 1))
    plt.savefig(os.path.join(os.path.join('log', args.model_name, file_name)))
    plt.close()


def main(args):
    dataset_path = get_dataset_path('lumi', 'dset')

    model = ROSTrajectoryVAE(args.model_name, args.latent_size, args.num_actions, num_joints=args.num_joints)
    vae = model.model
    loader = TrajectoryLoader(100, 16, dataset_path, actions_per_trajectory=args.num_actions)
    iter = loader.get_iterator(True)

    latents = []

    targets = []
    reconstructions = []

    for b in iter:
        trajectory = vae.to_torch(b)
        x = Variable(trajectory).to(vae.device)
        recon,  latent, _ = vae._forward(x, False)
        latent = latent.cpu().detach().numpy()
        latents.append(latent.transpose(1, 0))
        targets.append(vae.to_trajectory(trajectory.cpu().detach().numpy()))
        reconstructions.append(vae.to_trajectory(recon.cpu().detach().numpy()))


    # Latent distribution
    if args.latent:

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

    if args.trajectories:
        # Trajectory distributions
        trajectory_distributions(targets[0], reconstructions[0], "distributions_{}.png".format(0), args)


parser = argparse.ArgumentParser(description='Model debugger: latents distribution')
parser.add_argument('--latent-size', default=5, type=int, help='Number of latent variables')
parser.add_argument('--num-joints', default=7, type=int)
parser.add_argument('--num-actions', default=24, type=int)
parser.add_argument('--model-name', default="mse_fc_v1", type=str)
parser.add_argument('--no-latent', dest='latent',  action="store_false")
parser.set_defaults(debug=True)
parser.add_argument('--trajectories', action="store_true")

if __name__ == '__main__':
    main(parser.parse_args())
