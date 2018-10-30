import os
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
from utils import smooth_trajectory, MAX_ANGLE, MIN_ANGLE


class TrajectoryDataset(data.Dataset):

    def __init__(self, load_path, file_name, num_joints, actions_per_trajectory,  normalize=True, debug=False):
        self.load_path = load_path
        self.num_joints = num_joints
        self.num_actions = actions_per_trajectory
        self.file_name = file_name

        time_steps_raw, positions_raw, _, _ = self.load_trajectories(load_path, debug)

        self.num_samples =  time_steps_raw.shape[0]
        self.positions = np.zeros([self.num_samples, self.num_actions, self.num_joints])
        self.time_steps = np.zeros([self.num_samples, self.num_actions])
        self.process_trajectories(time_steps_raw, positions_raw, normalize)


    def __getitem__(self, index):
        return  torch.from_numpy(self.positions[index]).float()

    def __len__(self):
        return self.num_samples

    def load_trajectories(self, load_path, debug):

        if debug:
            time_steps_raw, positions_raw, velocity_raw, accelrator_raw = np.load(os.path.join(load_path, 'debug_trajectories.pkl'))
        else:
            time_steps_raw, positions_raw, velocity_raw, accelrator_raw = np.load(os.path.join(load_path, '{}.pkl'.format(self.file_name)))
        return time_steps_raw, positions_raw, velocity_raw, accelrator_raw

    def process_trajectories(self, time_steps_raw, positions_raw, normalize):

        for i in range(self.num_samples):
            smooth_steps, smooth_positions, _, _ = smooth_trajectory(time_steps_raw[i], positions_raw[i], self.num_actions, self.num_joints)
            self.time_steps[i] = smooth_steps
            self.positions[i] = smooth_positions

        if normalize:
            self.positions = (self.positions - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)

class TrajectoryLoader(object):

    def __init__(self, batch_size, num_processes, file_name, actions_per_trajectory=20, normalize=True, debug=False):

        dataset = TrajectoryDataset('/home/aleksi/hacks/behavioural_ws/trajectories', file_name, 7, actions_per_trajectory, normalize=normalize, debug=debug)
        train_size = int(dataset.__len__() * 0.7)
        test_size = dataset.__len__() - train_size
        trainset, testset = torch.utils.data.random_split(dataset.positions, (train_size, test_size))
        self.dataset = dataset # debig
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.trainset = trainset
        self.testset = testset

    def get_iterator(self, train):

        if train:
            dataset = self.trainset
        else:
            dataset = self.testset

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_processes)

if __name__ == '__main__':
    loader = TrajectoryLoader(10, 4, normalize=False, debug=True)
