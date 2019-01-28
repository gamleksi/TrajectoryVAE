import os
import torch
from model import TrajectoryVAE

ABSOLUTE_DIR = os.path.dirname(os.path.abspath(__file__))


def take_num(elem):
    elem = elem.split('_')[-1]
    elem = elem.split('.')[0]
    val = int(elem)
    return val

def model_name_search(folder_path, model_index=0):

    splitted = []

    for file in os.listdir(folder_path):
        if file.endswith(".tar"):
            splitted.append(file.split('.pth', 1)[0])
    assert(splitted.__len__() > 0)

    splitted.sort(key=take_num)
    return splitted[model_index]

class ROSTrajectoryVAE(object):

    def __init__(self, model_dir, latent_dim, num_actions, model_index=0, num_joints=7):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works for behavioral!')
        else:
            print('Behavioural is not using GPU')

        self.model = TrajectoryVAE(latent_dim, num_actions, num_joints, device, conv_model=False).to(device)
        self.load_parameters(model_dir, model_index)

    def load_parameters(self,model_dir, model_index):

        model_name = model_name_search(model_dir, model_index=model_index)
        path = os.path.join(model_dir, '{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def _process_sample(self, sample):
        sample = torch.FloatTensor(sample)
        return torch.unsqueeze(sample, 0)

    def get_result(self, sample):
        # Reconstructs a given sample
        sample = self._process_sample(sample)
        recon, latent = self.model.reconstruct(sample)
        return recon[0].detach().cpu().numpy(), latent[0].detach().cpu().numpy()

    def decode(self, sample):
        sample = self._process_sample(sample)
        recon = self.model.decode(sample)
        return recon[0].detach().cpu().numpy()


class RosTrajectoryConvVAE(ROSTrajectoryVAE):

    def __init__(self, model_folder, latent_size, num_actions, kernel_row, conv_channel, num_joints=7,  root_path=ABSOLUTE_DIR):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works!')
        else:
            print('YOU ARE NOT USING GPU')

        self.model = TrajectoryVAE(latent_size, num_actions, num_joints, device,
                          conv_model=True, kernel_row=kernel_row, conv_channel=conv_channel).to(device)

        self.load_parameters(model_folder, root_path)

def main():
    return ROSTrajectoryVAE("mse_v2", 4, 24, num_joints=7)

if __name__  == '__main__':
    model = main()
    print(model.decode([0.2] * 4).shape)
