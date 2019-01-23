import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np


class TrajectoryVisualizer(object):

    def __init__(self, sample_path):

        self.sample_path = sample_path
        self.create_path(self.sample_path)
        self.train_losses = []
        self.val_losses = []
        self.klds_train = []
        self.klds_val = []
        self.mses_train = []
        self.mses_val = []

    def create_path(self, path):
        if not(os.path.exists(path)):
            os.makedirs(path)

    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(np.log(train_loss))
        self.val_losses.append(np.log(val_loss))
        steps = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(steps, self.train_losses, 'r', label='Train')
        plt.plot(steps, self.val_losses, 'b', label='Validation')
        plt.title('Average Loss (in log scale)')
        plt.legend()
        plt.savefig(os.path.join(self.sample_path, 'log_loss.png'))
        plt.close()

    def update_klds(self, kld_train, kld_val):

        self.klds_train.append(np.log(kld_train))
        self.klds_val.append(np.log(kld_val))

        steps = range(1, len(self.klds_train) + 1)
        plt.figure()
        plt.plot(steps, self.klds_train, 'r', label='Train')
        plt.plot(steps, self.klds_val, 'b', label='Validation')

        plt.title('Average KLD (in log scale)')
        plt.legend()
        plt.savefig(os.path.join(self.sample_path, 'log_klds.png'))
        plt.close()

    def update_mses(self, mse_train, mse_val):

        self.mses_train.append(np.log(mse_train))
        self.mses_val.append(np.log(mse_val))

        steps = range(1, len(self.mses_train) + 1)
        plt.figure()
        plt.plot(steps, self.mses_train, 'r', label='Train')
        plt.plot(steps, self.mses_val, 'b', label='Validation')

        plt.title('Average MSE (in log scale)')
        plt.legend()
        plt.savefig(os.path.join(self.sample_path, 'log_mses.png'))
        plt.close()

    def generate_image(self, original, reconstructed, file_name=None, folder=None):
        fig = plt.figure(figsize=(30, 30))
        columns = 4
        rows = 1
        original = original.transpose(1, 0)
        reconstructed = reconstructed.transpose(1, 0)
        images = (original, reconstructed, np.abs(
            original - reconstructed))
        for i in range(1, 4):
            fig.add_subplot(rows, columns, i)
            im = plt.imshow(images[i-1],
                            cmap='gray', vmin=0, vmax=1)
        plt.colorbar(im, fig.add_subplot(rows, columns, 4))
        if file_name is None:
            plt.show()
        else:
            if folder is not None:
                path = os.path.join(self.sample_path, folder)
                self.create_path(path)
            else:
                path = self.sample_path
            plt.savefig(os.path.join(path, '{}.png'.format(file_name)))
        plt.close()

    def plot_trajectory(self, original, reconstructed, std_reconstructions=None, file_name=None, folder=None):

        fig = plt.figure(figsize=(30, 30))
        columns = 1
        rows = original.shape[0]
        steps = range(1, original.shape[1] + 1)
        for i in range(rows):

            fig.add_subplot(rows, columns, i + 1)
            plt.plot(steps, original[i], 'rs', label='Original joint {}'.format(i + 1))

            if std_reconstructions is None:
                plt.plot(steps, reconstructed[i], 'b^', label='Decoded joint {}'.format(i + 1))
            else:
                plt.plot(steps, reconstructed[i], 'b^', label='Decoded joint {}'.format(i + 1))
                plt.plot(steps, std_reconstructions[0][i], 'b--', linewidth=2.0, label='Negative std joint {}'.format(i + 1))
                plt.plot(steps, std_reconstructions[1][i], 'b--', linewidth=2.0, label='Positive std joint {}'.format(i + 1))

            plt.legend()
            plt.ylim(-0.1, 1.2)

        if file_name is None:
            plt.show()
        else:
            if folder is not None:
                path = os.path.join(self.sample_path, folder)
                self.create_path(path)
            else:
                path = self.sample_path
            plt.savefig(os.path.join(path, '{}.png'.format(file_name)))
        plt.close()

    def scatter_end_effoctor_poses(self, x_poses, y_poses):
        fig = plt.figure(figsize=(30, 30))
        plt.scatter(x_poses, y_poses)
        plt.close()
        plt.savefig(os.path.join(self.sample_path, '{}.png'.format('poses')))

    def trajectory_distributions(self, targets, reconstructions, file_name, folder=None):

        assert(targets.shape[0] == reconstructions.shape[0])

        num_joints = targets.shape[1]
        fig, axes = plt.subplots(num_joints, 2, sharex=True, sharey=True, figsize=[30, 30])
        steps = range(1, targets[0].shape[1] + 1)

        labels = ("targets", "reconstructed")

        for idx, trajectories in enumerate((targets, reconstructions)):
            for joint_idx in range(num_joints):
                ax = axes[joint_idx][idx]
                for traj_idx in range(len(targets)):
                    trajectory = trajectories[traj_idx]
                    ax.plot(steps, trajectory[joint_idx])
                ax.set_title("{} Joint {}".format(labels[idx], idx + 1))

        fig.tight_layout(pad=2)

        if folder is not None:
            path = os.path.join(self.sample_path, folder)
            self.create_path(path)
        else:
            path = self.sample_path


        plt.savefig(os.path.join(path, '{}.png'.format(file_name)))
        plt.close()

    def latent_distributions(self, latents, file_name, folder=None, bins=40):

        # latents = latents.transpose()

        fig, axes = plt.subplots(latents.shape[0], 1, sharex=True, figsize=[30, 30])
        for i in range(latents.shape[0]):
            ax = axes[i]
            batch = latents[i]
            ax.hist(batch, bins=bins)
            ax.set_title('Latent {}'.format(i + 1))
            ax.set_xlabel('x')
            ax.set_ylabel('frequency')

        fig.tight_layout(pad=2)
        if folder is not None:
            path = os.path.join(self.sample_path, folder)
            self.create_path(path)
        else:
            path = self.sample_path

        plt.savefig(os.path.join(path, '{}.png'.format(file_name)))
        plt.close()


import argparse

parser = argparse.ArgumentParser(description='Loss updated')
parser.add_argument('--folder-name', default='trajectory_test', type=str)

if __name__ == '__main__':

    args = parser.parse_args()
    model_path = os.path.join("log", args.folder_name)
    visualizer = TrajectoryVisualizer(model_path)

    log_file = os.path.join(model_path, 'log.csv')

    with open(log_file, 'r') as f:

        header = f.readline().rstrip()
        i = 0
        for line in f:
            line_strip = line.rstrip()
            components = line_strip.split(',')
            visualizer.train_losses.append(np.log(float(components[0])))
            visualizer.val_losses.append(np.log(float(components[1])))
            visualizer.mses_train.append(np.log(float(components[2])))
            visualizer.mses_val.append(np.log(float(components[3])))
            visualizer.klds_train.append(np.log(float(components[4])))
            visualizer.klds_val.append(np.log(float(components[5])))

        visualizer.update_losses(np.exp(visualizer.train_losses[-1]), np.exp(visualizer.val_losses[-1]))
        visualizer.update_klds(np.exp(visualizer.klds_train[-1]), np.exp(visualizer.klds_val[-1]))
        visualizer.update_mses(np.exp(visualizer.mses_train[-1]), np.exp(visualizer.mses_val[-1]))
