import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.nn import functional as F
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.random.manual_seed(555)
np.random.seed(444)

LUMI_X_LIM = [0.3, 0.55]
LUMI_Y_LIM = [-0.4, 0.4]

class Predictor(nn.Module):

    def __init__(self, latent_size):

        super(Predictor, self).__init__()

        self.fc1_mean = nn.Linear(2, 24)
        self.fc2_mean = nn.Linear(24, 24)
        self.fc3_mean = nn.Linear(24, latent_size)

#        self.fc1_scale = nn.Linear(3, 24)
#        self.fc2_scale = nn.Linear(24, 24)
#        self.fc3_scale = nn.Linear(24, latent_size)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        self.init_weights()

    def print_model(self):

        print(self.fc1_mean.weight)
        print(self.fc1_mean.bias)
        print(self.fc2_mean.weight)
        print(self.fc2_mean.bias)
        print(self.fc3_mean.weight)
        print(self.fc3_mean.bias)

    def forward(self, x):

        loc = self.relu(self.fc1_mean(x))
        loc = self.relu(self.fc2_mean(loc))
        loc = self.fc3_mean(loc)

#        scale = self.relu(self.fc1_scale(x))
#        scale = self.relu(self.fc2_scale(scale))
#        scale = self.softplus(self.fc3_scale(scale))

        return loc

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-2)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


def plot_loss(train, val, title, save_to):

    steps = range(1, train.__len__() + 1)

    fig = plt.figure()

    plt.plot(steps, train, 'r', label='Train')
    plt.plot(steps, val, 'b', label='Validation')

    plt.title(title)
    plt.legend()
    plt.savefig(save_to)
    plt.close()


def get_model_directory(model):
    return os.path.join('log', model)


def get_reconstruction_results(model, index):
    model_dir = get_model_directory(model)
    results_path = os.path.join(model_dir, 'reconstruction_results')
    results = os.listdir(results_path)
    results.sort() # TODO fix the bug
    print(results[index])
    end_poses, pose_results, latents, trajectories, recons = np.load(os.path.join(results_path , results[index]))
    return end_poses, pose_results, latents, trajectories, recons


def load_dataset(model, index, preprocess=True):
    _, coords, latents, _, _ = get_reconstruction_results(model, index)

    # Simplifies the dataset
    if (preprocess):
        x_limit = (LUMI_X_LIM[0] + 0.1 < coords[:, 0]) * (coords[:, 0] < LUMI_X_LIM[1])
        y_limit = (LUMI_Y_LIM[0] < coords[:, 1]) * (coords[:, 1] < LUMI_Y_LIM[1])
        limits = x_limit * y_limit
        coords = coords[limits]
        latents = latents[limits]

    X = torch.Tensor(coords)
    Y = torch.Tensor(latents)

    return data.TensorDataset(X[:, :2], Y)

def save_arguments(args, save_path):

    args = vars(args)

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    file = open(os.path.join(save_path, "arguments.txt"), 'w')
    lines = [item[0] + " " + str(item[1]) + "\n" for item in args.items()]
    file.writelines(lines)

def main(args):

    save_path = os.path.join('pred_log', args.save_folder)

    save_arguments(args, save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print('GPU works for behavioral!')
    else:
        print('Behavioural is not using GPU')

    model = Predictor(latent_size=args.latent_size)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    dataset = load_dataset(args.vae_name, args.model_index)

    print("Dataset size", dataset.__len__())
    train_size = int(dataset.__len__() * 0.7)
    test_size = dataset.__len__() - train_size

    trainset, testset = data.random_split(dataset, (train_size, test_size))

    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_processes)
    test_loader = data.DataLoader(testset, batch_size=testset.__len__())

    best_val = np.inf
    avg_train_losses = []
    avg_val_losses = []


    for epoch in range(args.num_epoch):

        print("Epoch {}".format(epoch + 1))

        model.train()

        # Training
        train_losses = []
        for coords, targets in tqdm(train_loader):

            coords, targets = coords.to(device), targets.to(device)

            outputs = model(Variable(coords))
            loss = F.mse_loss(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        avg_loss = np.mean(train_losses)
        avg_train_losses.append(avg_loss)
        print("Average training Loss {}".format(avg_loss))

        model.eval()

        # Validation
        val_losses = []
        for coords, targets in tqdm(test_loader):

            coords, targets = coords.to(device), targets.to(device)
            outputs = model(Variable(coords))
            loss = F.mse_loss(outputs, targets)

            val_losses.append(loss.item())

        avg_loss = np.mean(val_losses)
        avg_val_losses.append(avg_loss)

        print("Average validation Loss {}".format(avg_loss))

        if avg_loss < best_val:
            best_val = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth.tar'))

        plot_loss(avg_train_losses, avg_val_losses, 'Avg mse', os.path.join(save_path, 'avg_mse.png'))
        plot_loss(np.log(avg_train_losses), np.log(avg_val_losses), 'Avg mse in log scale', os.path.join(save_path, 'avg_log_mse.png'))

parser = argparse.ArgumentParser(description='Latent Predictor')

parser.add_argument('--latent-size', default=5, type=int, help='Number of latent variables')
parser.add_argument('--vae-name', default='rtt-star-beta-iterative-latent-5-b-max-small_v2', type=str)
parser.add_argument('--model-index', default=0, type=int)
parser.add_argument('--batch-size', default=124, type=int)
parser.add_argument('--num-processes', default=17, type=int)
parser.add_argument('--num-epoch', default=17, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--save-folder', default='example', type=str)


if __name__ == '__main__':
    main(parser.parse_args())
