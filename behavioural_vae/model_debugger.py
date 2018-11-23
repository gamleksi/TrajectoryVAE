import os

import numpy as np
from ros_monitor import ROSTrajectoryVAE
from torch.autograd import Variable
from trajectory_loader import TrajectoryLoader

import matplotlib.pyplot as plt

def get_dataset_path(folder_name, dataset_root):
    return os.path.join(dataset_root, folder_name, 'trajectories.pkl')

def main():
    num_actions = 24
    latent_size = 5
    num_joints = 7
    dataset_path = get_dataset_path('lumi', 'dset')
    batch_size = 785
    num_processes = 16

    model = ROSTrajectoryVAE("mse_fc_v1", latent_size, num_actions, num_joints=num_joints)
    vae = model.model
    loader = TrajectoryLoader(batch_size, num_processes, dataset_path, actions_per_trajectory=num_actions)
    iter = loader.get_iterator(True)

    latents = []

    for b in iter:
        trajectory = vae.to_torch(b)
        x = Variable(trajectory).to(vae.device)
        recon,  latent, _ = vae._forward(x, False)
        traj = vae.to_trajectory(recon).cpu().detach().numpy()
        latent = latent.cpu().detach().numpy()
        latents.append(latent.transpose(1, 0))

    latents = np.concatenate([l for l in latents], axis=1)
    fig, axes = plt.subplots(latent_size, 1, sharey=True, figsize=[30, 30])

    for i in range(latent_size):
        ax = axes[i]
        batch = latents[i]
        ax.hist(batch, 600)
        ax.set_title('Latent {}'.format(i + 1))
        ax.set_xlabel('x')
        ax.set_ylabel('frequency')

    fig.tight_layout(pad=2)
    plt.savefig(os.path.join('latents_fc_v1.png'))
    plt.close()


if __name__ == '__main__':
    main()
