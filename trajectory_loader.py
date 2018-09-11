import os
import numpy as np
import torch
from scipy import interpolate
from torchvision import transforms
import torch.utils.data as data

class TrajectoryDataset(data.Dataset):

    def __init__(self, load_path, num_joints, actions_per_trajectory, normalize=True, debug=False):
        self.load_path = load_path
        self.num_joints = num_joints
        self.num_actions = actions_per_trajectory

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
            time_steps_raw, positions_raw, velocity_raw, accelrator_raw = np.load(os.path.join(load_path, 'trajectories.pkl'))
        return time_steps_raw, positions_raw, velocity_raw, accelrator_raw

    def process_trajectories(self, time_steps_raw, positions_raw, normalize):

        for i in range(self.num_samples):
            smooth_steps, smooth_positions, _, _ = self.smooth_trajectory(time_steps_raw[i], positions_raw[i])
            self.time_steps[i] = smooth_steps
            self.positions[i] = smooth_positions

        if normalize:
            max_positions = np.max(self.positions.reshape(-1, self.num_joints), axis=0)
            min_positions = np.min(self.positions.reshape(-1, self.num_joints), axis=0)
            for i in range(self.num_samples):
                for j in range(self.num_actions):
                    self.positions[i, j] = (self.positions[i, j] - min_positions) / (max_positions - min_positions)

    def smooth_trajectory(self, time_steps_raw, positions_raw):

        duration = time_steps_raw[-1]
        num_samples_raw = positions_raw.shape[0]
        smooth_steps = np.linspace(0, duration, self.num_actions)

        spls = [interpolate.splrep(time_steps_raw, positions_raw[:,i])
                                               for i in range(self.num_joints)]

        smooth_positions = np.stack([interpolate.splev(smooth_steps, spls[i], der=0) for i in range(self.num_joints)]).T
        smooth_velocities = np.stack([interpolate.splev(smooth_steps, spls[i], der=1) for i in range(self.num_joints)]).T
        smooth_accelerations = np.stack([interpolate.splev(smooth_steps, spls[i], der=2) for i in range(self.num_joints)]).T

        return smooth_steps, smooth_positions, smooth_velocities, smooth_accelerations


class TrajectoryLoader(object):

    def __init__(self, batch_size, num_processes, actions_per_trajectory=20, normalize=False, debug=False):

        dataset = TrajectoryDataset('/home/aleksi/hacks/behavioural_ws/trajectories', 7, actions_per_trajectory, normalize=normalize, debug=debug)
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
    loader = TrajectoryLoader(10, 4)
