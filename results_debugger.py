import os

import numpy as np
from ros_monitor import ROSTrajectoryVAE
from torch.autograd import Variable
from trajectory_loader import TrajectoryLoader
import argparse
from visual import TrajectoryVisualizer
import torch


'''
To understand trade-off between reconstruction and KL-divergence 
'''


def get_dataset_path(folder_name, dataset_root):
    return os.path.join(dataset_root, folder_name, 'trajectories.pkl')


def get_number_of_models(model_name):

    splitted = []

    for file in os.listdir(os.path.join('log', model_name)):

        if file.endswith(".tar"):
            splitted.append(file.split('.pth', 1)[0])

    assert(splitted.__len__() > 0)

    num_models = len(splitted)
    return num_models


def main(args):

    dataset_path = get_dataset_path(args.dataset, 'dset')

    if args.debug:
        num_models = 1
    else:
        num_models = get_number_of_models(args.model_name)

    path = os.path.join('log', args.model_name)
    visualizer = TrajectoryVisualizer(path)
    loader = TrajectoryLoader(100, 16, dataset_path, actions_per_trajectory=args.num_actions)
    num_samples = loader.trainset.__len__()
    iter = loader.get_iterator(True)

    log_file = open(os.path.join(path, "debugger_log.txt"), 'w')
    log_file.write("Model Name {} '\n'".format(args.model_name))

    for model_idx in range(num_models):

        model = ROSTrajectoryVAE(args.model_name, args.latent_size, args.num_actions, model_index=model_idx, num_joints=args.num_joints)
        vae = model.model

        latents = []
        targets = []
        reconstructions = []
        std_reconstructions = []
        stds_sum = np.zeros(args.latent_size)
        mean_distances = np.zeros(args.num_joints)

        for b in iter:
            trajectory = vae.to_torch(b)
            x = Variable(trajectory).to(vae.device)
            recon, latent, log_var = vae._forward(x, False)
            std = torch.exp(0.5 * log_var)

            neg_recon = vae.decode(latent - std)
            pos_recon = vae.decode(latent + std)

            std = std.cpu().detach().numpy()
            stds_sum += std.sum(0)

            neg_recon = vae.to_trajectory(neg_recon.cpu().detach().numpy())
            pos_recon = vae.to_trajectory(pos_recon.cpu().detach().numpy())
            recon = vae.to_trajectory(recon.cpu().detach().numpy())

            latents.append(latent.cpu().detach().numpy().transpose(1, 0))
            targets.append(vae.to_trajectory(trajectory.cpu().detach().numpy()))
            reconstructions.append(recon)
            errors = (neg_recon, pos_recon)

            std_reconstructions.append(errors)

            distances = np.abs(recon - neg_recon) + np.abs(recon - pos_recon)
            mean_distances += distances.mean(axis=(0, 2))

        std_mean = stds_sum / num_samples
        print("Model", model_idx)
        print("STD mean", std_mean)
        print("Mean distances", mean_distances)
        log_file.write("id {} '\n'".format(model_idx))
        log_file.write("STD mean {} '\n'".format(std_mean))
        log_file.write("Average of STD means {} '\n'".format(std_mean.mean()))

        log_file.write("Mean distances {} '\n'".format(mean_distances))
        log_file.write("Average of distance means {} '\n'".format(mean_distances.mean()))

        # Latent distribution
        if args.latent:
            latents = np.concatenate([l for l in latents], axis=1)
            visualizer.latent_distributions(latents, 'latent_distribution_model_{}.png'.format(model_idx), folder='results', bins=600)

        if args.trajectories:
            # Trajectory distributions
            for idx in range(3):
                visualizer.trajectory_distributions(targets[idx], reconstructions[idx], "distributions_{}_model_{}.png".format(model_idx, idx), folder='results')

        if args.std_trajectories:

            for idx in range(10):
                visualizer.plot_trajectory(
                    targets[idx][0], reconstructions[idx][0],
                    std_reconstructions=std_reconstructions[idx][0],
                    file_name="std_samples_{}_model_{}.png".format(idx, model_idx), folder='results')
    log_file.close()


parser = argparse.ArgumentParser(description='Model debugger: latent distribution')
parser.add_argument('--latent-size', default=5, type=int, help='Number of latent variables')
parser.add_argument('--num-joints', default=7, type=int)
parser.add_argument('--num-actions', default=24, type=int)
parser.add_argument('--model-name', default="mse_fc_v1", type=str)
parser.add_argument('--no-latent', dest='latent',  action="store_false")
parser.add_argument('--trajectories', action="store_true")
parser.add_argument('--dataset', default="lumi_rtt_star", type=str)
parser.add_argument('--std-trajectories', action="store_true")
parser.add_argument('--debug', action="store_true")

if __name__ == '__main__':
    main(parser.parse_args())
