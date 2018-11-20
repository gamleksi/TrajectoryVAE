import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, num_actions, num_joints, latent_size, kernel_row=4, channel_out=2):
        super(Encoder, self).__init__()

        assert(num_actions % kernel_row == 0) # "Action steps is required to be dividable by kernel row."

        self.conv1 = nn.Conv1d(num_joints, channel_out, kernel_row)

        self.conv_out = (num_actions - kernel_row + 1) * channel_out

        self.fc1 = nn.Linear(self.conv_out, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 250)

        self.fc4_loc = nn.Linear(250, latent_size)
        self.fc4_scale = nn.Linear(250, latent_size)

        self.bn1 = nn.BatchNorm1d(channel_out)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(250)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, self.conv_out)
        x = self.relu(self.bn2(self.fc1(x)))
        x = self.relu(self.bn3(self.fc2(x)))
        x = self.relu(self.bn4(self.fc3(x)))

        loc = self.fc4_loc(x)
        scale = self.fc4_scale(x)

        return loc, scale


class Decoder(nn.Module):
    def __init__(self, num_actions, num_joints, latent_size, kernel_row=4, channel_in=2):

        super(Decoder, self).__init__()

        assert(num_actions % kernel_row == 0) # "Action steps is required to be dividable by kernel row."

        self.conv_out = (num_actions - kernel_row + 1) * channel_in
        self.channel_in = channel_in

        self.fc1 = nn.Linear(latent_size, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, self.conv_out)
        self.t_conv1 = nn.ConvTranspose1d(channel_in, num_joints, kernel_row)

        self.bn1 = nn.BatchNorm1d(250)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(self.conv_out)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = x.view(x.shape[0], self.channel_in, self.conv_out / self.channel_in)
        x = self.sigmoid(self.t_conv1(x))
        return x

def main():

    num_actions = 64
    num_joints = 7
    latent_size = 5
    encoder = Encoder(num_actions, num_joints, latent_size)
    encoder.train()
    x = torch.zeros(5, num_joints, num_actions)
    x = Variable(x)
    loc, scale = encoder(x)
    decoder = Decoder(num_actions, num_joints, latent_size)
    res = decoder(loc)
    return res.shape == x.shape

if __name__ ==  "__main__":

    print(main())



