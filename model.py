import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)

        self.fc4_loc = nn.Linear(250, output_size)
        self.fc4_scale = nn.Linear(250, output_size)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(250)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))

        loc = self.fc4_loc(x)
        scale = self.fc4_scale(x)

        return loc, scale

class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, output_size)

        self.bn1 = nn.BatchNorm1d(250)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(1000)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))

        return x

from torch.autograd import Variable
from torch.nn import functional as F

class TrajectoryVAE(nn.Module):

    def __init__(self, NUM_LATENT_VARIABLES, num_actions, num_joints, device, beta=1):
        encoder = Encoder(num_actions *num_joints, NUM_LATENT_VARIABLES)
        decoder = Decoder(NUM_LATENT_VARIABLES, num_actions * num_joints)
        super(TrajectoryVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_actions = num_actions
        self.num_joints = num_joints
        self.beta = beta

    def set_mode(self, train):
        if train:
            self.train()
        else:
            self.eval()

    def _forward(self, x, train):
        self.set_mode(train)
        mu, logvar  = self.encoder(x)
        z = self._reparameterize(mu, logvar, train)
        return self.decoder(z), mu, logvar

    def _reparameterize(self, mu, logvar, train):
        if train:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(self.device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def to_vector(self, trajectory):
        return trajectory.reshape([trajectory.shape[0], self.num_actions * self.num_joints]).to(self.device).float()

    def to_trajectory(self, vec):
        return vec.reshape([vec.shape[0], self.num_actions, self.num_joints])

    def evaluate(self, state):
        # state includes batch samples and a train / test flag
        # samples should be tensors and processed in loader function.
        assert(len(state[0].shape) == 3)
        trajectories = self.to_vector(state[0])

        x = Variable(trajectories)
        train = state[1]
        x_recon,  mu, log_var = self._forward(x, train)

        BCE = F.binary_cross_entropy(x_recon, trajectories, size_average=False)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + self.beta * KLD, self.to_trajectory(x_recon)

    def latent_distribution(self, sample):
        self.set_mode(False)
        x = Variable(sample.to(self.device))
        mu, logvar = self.encoder(x)

        return mu, logvar

    def reconstruct(self, sample):
        x = Variable(sample).to(self.device)
        recon,  _, _ = self._forward(x, False)

        return self.to_trajectory(recon)
