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

    def create_path(self, path):
        if not(os.path.exists(path)):
            os.makedirs(path)

    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        steps = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(steps, self.train_losses, 'r', label='Train')
        plt.plot(steps, self.val_losses, 'b', label='Validation')
        plt.title('Average Loss')
        plt.legend()
        plt.savefig(os.path.join(self.sample_path, 'loss.png'))
        plt.close()

    def generate_image(self, original, reconstructed, file_name=None, folder=None):
        # w = original.shape[0]
        # h = original.shape[1]
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

    def plot_trajectory(self, original, reconstructed, file_name=None, folder=None):
        fig = plt.figure(figsize=(30, 30))
        columns = 1
        original = original.transpose(1, 0)
        reconstructed = reconstructed.transpose(1, 0)
        rows = original.shape[1]
        steps = range(1, original.shape[0] + 1)
        for i in range(rows):
            fig.add_subplot(rows, columns, i + 1)
            plt.plot(steps, original[:, i], 'ro', label='Original joint {}'.format(i + 1))
            plt.plot(steps, reconstructed[:, i], 'bo', label='Decoded joint {}'.format(i + 1))
            plt.legend()

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

if __name__ == '__main__':
    TrajectoryVisualizer(os.path.join('log', 'lumi_v3'))
