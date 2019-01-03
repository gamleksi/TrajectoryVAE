import numpy as np
import torch
import torch.utils.data as data
from utils import smooth_trajectory, MAX_ANGLE, MIN_ANGLE


class TrajectoryDataset(data.Dataset):

    def __init__(self, file_path, num_joints, actions_per_trajectory):

        self.num_joints = num_joints
        self.num_actions = actions_per_trajectory

        time_steps_raw, positions_raw, _, _ = self.load_trajectories(file_path)

        self.num_samples =  time_steps_raw.shape[0]
        self.positions = np.zeros([self.num_samples, self.num_joints, self.num_actions])
        self.time_steps = np.zeros([self.num_samples, self.num_actions])
        self.process_trajectories(time_steps_raw, positions_raw)

    def __getitem__(self, index):
        return  torch.from_numpy(self.positions[index]).float()

    def __len__(self):
        return self.num_samples

    def load_trajectories(self, file_path):

        time_steps_raw, positions_raw, velocity_raw, accelrator_raw = np.load(file_path)
        return time_steps_raw, positions_raw, velocity_raw, accelrator_raw

    def process_trajectories(self, time_steps_raw, positions_raw):

        for i in range(self.num_samples):
            smooth_steps, smooth_positions, _, _ = smooth_trajectory(time_steps_raw[i], positions_raw[i], self.num_actions, self.num_joints)
            self.time_steps[i] = smooth_steps
            self.positions[i] = smooth_positions

        self.positions = (self.positions - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)

class TrajectoryLoader(object):

    def __init__(self, batch_size, num_processes, file_path, actions_per_trajectory=20, num_joints=7):

        self.dataset = TrajectoryDataset(file_path, num_joints, actions_per_trajectory)
        train_size = int(self.dataset.__len__() * 0.9)
        test_size = self.dataset.__len__() - train_size
        trainset, testset = torch.utils.data.random_split(self.dataset.positions, (train_size, test_size))
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.trainset = trainset
        self.testset = testset
        self.visualset = testset[:100]

    def get_iterator(self, train):

        if train:
            dataset = self.trainset
        else:
            dataset = self.testset

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_processes)

    def visual_trajectories(self):
        return torch.from_numpy(self.visualset)

if __name__ == '__main__':

    loader = TrajectoryLoader(10, 4,
                              '/home/aleksi/mujoco_ws/src/motion_planning/trajectory_data/example/trajectories.pkl')
    loader.dataset.__getitem__(0)
    visuals = loader.visual_trajectories()
    print(visuals.shape)

