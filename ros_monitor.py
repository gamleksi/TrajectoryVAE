import os
import torch
from model import TrajectoryVAE
#from blender_dataset import KinectImageHandler
import numpy as np

ABSOLUTE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_name_search(folder_path):

  for file in os.listdir(folder_path):
        if file.endswith(".tar"):
          splitted = file.split('.pth', 1)

  return splitted[0]

class ROSTrajectoryVAE(object):

    def __init__(self, model_folder, latent_dim, num_actions, num_joints=7, root_path=ABSOLUTE_DIR):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works!')
        else:
            print('YOU ARE NOT USING GPU')

        self.model = TrajectoryVAE(latent_dim, num_actions, num_joints, device).to(device)
        self.load_parameters(model_folder, root_path)

    def load_parameters(self, folder, root_path):
        model_path = os.path.join(root_path, 'log', folder)
        model_name = model_name_search(model_path)
        path = os.path.join(model_path, '{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def _process_sample(self, sample):
        sample = torch.FloatTensor(sample)
        return torch.unsqueeze(sample, 0)

    def get_result(self, sample):
        sample = self._process_sample(sample)
        recon = self.model.reconstruct(sample)
        return recon[0].detach().cpu().numpy()

def main():
    return ROSTrajectoryVAE('no_normalization_model', 10, 20)

if __name__  == '__main__':
    model = main()
