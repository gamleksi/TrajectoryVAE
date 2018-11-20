import numpy as np
import torch
from visual import TrajectoryVisualizer
import torchnet as tnt
from torchnet.engine import Engine
import csv
import os


class Trainer(Engine):

    def __init__(self, dataloader, model, save_folder=None, save_name=None, log=False, debug=False):
        super(Trainer, self).__init__()

        self.debug = debug
        self.dataloader = dataloader
        self.get_iterator = dataloader.get_iterator
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.initialize_engine()

        self.model = model
        self.log_data = log
        self.save_folder = save_folder

        if self.log_data:
            assert(save_folder is not None and save_name is not None)
            self.initilize_log(save_folder, save_name)
            self.best_loss = np.inf
            self.visualizer = TrajectoryVisualizer(os.path.join("log", self.save_folder))

    def initialize_engine(self):
        self.hooks['on_sample'] = self.on_sample
        self.hooks['on_forward'] = self.on_forward
        self.hooks['on_start_epoch'] = self.on_start_epoch
        self.hooks['on_end_epoch'] = self.on_end_epoch

    def train(self, num_epoch, optimizer):
        super(Trainer, self).train(self.model.evaluate, self.get_iterator(True), maxepoch=num_epoch, optimizer=optimizer)

    def reset_meters(self):
        self.meter_loss.reset()

    def on_sample(self, state):
        state['sample'] = (state['sample'], state['train'])

    def on_forward(self, state):
        loss = state['loss']
        self.meter_loss.add(loss.item())

    def on_start_epoch(self, state):
        self.model.new_epoch()
        self.model.train(True)
        self.reset_meters()

    def visual_trajectories(self, epoch):
        trajectories = self.dataloader.visual_trajectories().float()
        results = self.model.reconstruct(trajectories)
        results = results.detach().cpu()
        folder = "trajectories_{}".format(epoch)
        for i in range(results.shape[0]):
            self.visualizer.generate_image(trajectories[i], results[i], file_name="{}_image".format(i), folder=folder)
            self.visualizer.plot_trajectory(trajectories[i].numpy(), results[i].numpy(), file_name="{}_trajectory".format(i), folder=folder)

    def on_end_epoch(self, state):

        epoch = int(state['epoch'])
        train_loss = self.meter_loss.value()[0]
        print("EPOCH: {}".format(epoch))
        print("Avg Training loss: {}".format(train_loss))
        self.reset_meters()
        self.test(self.model.evaluate, self.get_iterator(False))
        val_loss = self.meter_loss.value()[0]
        print("Avg Validation loss: {}".format(val_loss))

        if self.log_data:
            self.visualizer.update_losses(train_loss, val_loss)
            self.log_csv(train_loss, val_loss, val_loss < self.best_loss)

            if val_loss < self.best_loss:
                self.save_model()
                self.best_loss = val_loss

            if epoch % 25 == 0:
                self.visual_trajectories(epoch)

    def initilize_log(self, save_folder, save_name):

        self.log_path = 'log/{}'.format(save_folder)

        if not(self.debug):
            assert(not(os.path.exists(self.log_path))) # remove a current folder with the same name or rename the suggested folder
            os.makedirs(self.log_path)
        elif not(os.path.exists(self.log_path)):
            os.makedirs(self.log_path)

        self.csv_path = os.path.join(self.log_path, 'log_{}.csv'.format(save_name))
        self.model_path = os.path.join(self.log_path, '{}.pth.tar'.format(save_name))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def log_csv(self, train_loss, val_loss, improved):

        fieldnames = ['train_loss', 'val_loss', 'improved']
        fields = [train_loss, val_loss, int(improved)]

        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            if not(file_exists):
                writer.writeheader()
            row = {}
            for i, name in enumerate(fieldnames):
                row[name] = fields[i]

            writer.writerow(row)

