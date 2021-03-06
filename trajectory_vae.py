import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import conv_model as cm


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


def load_parameters(model, model_dir, model_index):

    model_name = model_name_search(model_dir, model_index=model_index)
    path = os.path.join(model_dir, '{}.pth.tar'.format(model_name))
    model.load_state_dict(torch.load(path))
    model.eval()


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 250)

        self.fc5_loc = nn.Linear(250, output_size)
        self.fc5_scale = nn.Linear(250, output_size)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(250)

    def forward(self, x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))

        loc = self.fc5_loc(x)
        scale = self.fc5_scale(x)

        return loc, scale


class SimpleEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleEncoder, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 250)

        self.fc4_loc = nn.Linear(250, output_size)
        self.fc4_scale = nn.Linear(250, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        loc = self.fc4_loc(x)
        scale = self.fc4_scale(x)

        return loc, scale


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 1000)
        self.fc5 = nn.Linear(1000, output_size)

        self.bn1 = nn.BatchNorm1d(250)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(1000)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.sigmoid(self.fc5(x))
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleDecoder, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x



class TrajectoryVAE(nn.Module):

    def __init__(self, latent_size, num_actions,  num_joints, device, num_epoch=100,
                 conv_model=True, kernel_row=4, conv_channel=2, beta_interval=25, beta_min=1.0e-4, beta_max=1.0e-0):

        self.conv_model = conv_model

        if self.conv_model:
            encoder = cm.Encoder(num_actions, num_joints, latent_size, kernel_row=kernel_row, channel_out=conv_channel)
            decoder = cm.Decoder(num_actions, num_joints, latent_size, kernel_row=kernel_row, channel_in=conv_channel)
        else:
            encoder = SimpleEncoder(num_actions *num_joints, latent_size)
            decoder = SimpleDecoder(latent_size, num_actions * num_joints)

        super(TrajectoryVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_actions = num_actions
        self.num_joints = num_joints
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta = self.beta_min
        self.epoch_max = num_epoch
        self.current_epoch = 0
        self.beta_interval = beta_interval

    def set_mode(self, train):
        if train:
            self.train()
        else:
            self.eval()

    def _forward(self, x, train):

        self.set_mode(train)
        mu, logvar = self.encoder(x)
        z = self._reparametrize(mu, logvar, train)
        return self.decoder(z), mu, logvar

    def _reparametrize(self, mu, logvar, train):

        if train:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(self.device)
            return eps.mul(std).add(mu)
        else:
            return mu

    def to_torch(self, trajectories):
        assert(len(trajectories.shape) == 3)
        if self.conv_model:
            return trajectories.to(self.device).float()
        else:
            return trajectories.reshape([trajectories.shape[0], self.num_actions * self.num_joints]).to(self.device).float()

    def to_trajectory(self, vec):
        return vec.reshape([vec.shape[0], self.num_joints, self.num_actions])

    def new_epoch(self):
        self.current_epoch += 1

    def beta_updated(self):
        return self.current_epoch % self.beta_interval == 0


    def evaluate(self, state):

        # state includes batch samples and a train / test flag
        trajectories = self.to_torch(state[0])
        x = Variable(trajectories)
        train = state[1]
        x_recon,  mu, log_var = self._forward(x, train)

        renonstruction_loss = F.mse_loss(x_recon, trajectories)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - 0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), 1)
        KLD = KLD.mean()

        if self.beta_updated():
            self.beta = self.beta_min + 1.0 * self.current_epoch * (self.beta_max - self.beta_min) / self.epoch_max

        return renonstruction_loss + self.beta * KLD, (renonstruction_loss, KLD)

    def latent_distribution(self, sample):
        self.set_mode(False)
        x = Variable(sample.to(self.device))
        mu, logvar = self.encoder(x)

        return mu, logvar

    def decode(self, sample):
        assert(len(sample.shape) == 2)
        self.set_mode(False)
        z = Variable(sample.to(self.device))
        return self.to_trajectory(self.decoder(z))

    def reconstruct(self, sample):
        trajectory = self.to_torch(sample)
        x = Variable(trajectory).to(self.device)
        recons, latents, _ = self._forward(x, False)
        return self.to_trajectory(recons), latents
