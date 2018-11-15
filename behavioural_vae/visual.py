import os
import matplotlib.pyplot as plt
import numpy as np


class TrajectoryVisualizer(object):

    def __init__(self, sample_path):

        self.sample_path = sample_path
        if not(os.path.exists(self.sample_path)):
            os.makedirs(self.sample_path)

    def generate_image(self, original, reconstructed, file_name=None):
        # w = original.shape[0]
        # h = original.shape[1]
        fig = plt.figure(figsize=(30, 30))
        columns = 4
        rows = 1
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
            plt.savefig(os.path.join(self.sample_path, '{}.png'.format(file_name)))
            plt.close()

if __name__ == '__main__':
    TrajectoryVisualizer(os.path.join('log', 'lumi_v3'))