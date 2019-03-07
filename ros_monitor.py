import torch
from trajectory_vae import TrajectoryVAE, load_parameters


class ROSTrajectoryVAE(object):

    def __init__(self, model_dir, latent_dim, num_actions, model_index=0, num_joints=7):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works for behavioral!')
        else:
            print('Behavioural is not using GPU')

        self.model = TrajectoryVAE(latent_dim, num_actions, num_joints, device, conv_model=False).to(device)
        load_parameters(self.model, model_dir, model_index)

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

    def __init__(self, model_dir, latent_size, num_actions, kernel_row, conv_channel, model_index=0, num_joints=7):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works!')
        else:
            print('YOU ARE NOT USING GPU')

        self.model = TrajectoryVAE(latent_size, num_actions, num_joints, device,
                          conv_model=True, kernel_row=kernel_row, conv_channel=conv_channel).to(device)

        load_parameters(self.model, model_dir, model_index)

def main():
    return ROSTrajectoryVAE("mse_v2", 4, 24, num_joints=7)

if __name__  == '__main__':
    model = main()
    print(model.decode([0.2] * 4).shape)
