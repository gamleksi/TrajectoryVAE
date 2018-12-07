import numpy as np
import torch
from visual import TrajectoryVisualizer
import torchnet as tnt
from torchnet.engine import Engine
import csv
import os

class Saver(object):

    def __init__(self, model_name, save_path):
        self.model_name = model_name
        self.save_path = save_path
        self.beta_update = 0

    def update_beta(self, beta_updated):
        if beta_updated:
            self.beta_update += 1

    def log_csv(self, train_loss, val_loss, mse_loss, mse_val, kld_train, kld_val, improved):

        fieldnames = ['train_loss', 'val_loss', 'mse_loss', 'mse_val', 'kld_train', 'kld_val', 'improved', 'beta_update']
        fields = [train_loss, val_loss, mse_loss, mse_val, kld_train, kld_val, int(improved), self.beta_update]
        csv_path = os.path.join(self.save_path, 'log.csv')
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            if not(file_exists):
                writer.writeheader()
            row = {}
            for i, name in enumerate(fieldnames):
                row[name] = fields[i]

            writer.writerow(row)

    def save_model(self, model):

        model_path = os.path.join(self.save_path, '{}_iter_{}.pth.tar'.format(self.model_name, self.beta_update))
        torch.save(model.state_dict(), model_path)

class Trainer(Engine):

    def __init__(self, dataloader, model, save_folder=None, save_name=None, log=False, debug=False):
        super(Trainer, self).__init__()

        self.debug = debug
        self.dataloader = dataloader
        self.get_iterator = dataloader.get_iterator
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.kld = tnt.meter.AverageValueMeter()
        self.mse = tnt.meter.AverageValueMeter()

        self.initialize_engine()

        self.model = model
        self.log_data = log
        self.save_folder = save_folder

        if self.log_data:
            assert(save_folder is not None and save_name is not None)
            self.saver = Saver(save_name, os.path.join('log', self.save_folder))
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
        self.kld.reset()
        self.mse.reset()

    def on_sample(self, state):
        state['sample'] = (state['sample'], state['train'])

    def on_forward(self, state):
        loss = state['loss']
        mse, KLD = state['output']
        self.meter_loss.add(loss.item())
        self.mse.add(mse.item())
        self.kld.add(KLD.item())

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
        print("EPOCH: {}".format(epoch))
        train_loss = self.meter_loss.value()[0]
        mse_train = self.mse.value()[0]
        kld_train = self.kld.value()[0]

        self.reset_meters()
        self.test(self.model.evaluate, self.get_iterator(False))
        val_loss = self.meter_loss.value()[0]
        mse_val = self.mse.value()[0]
        kld_val = self.kld.value()[0]

        print("Loss train: {}, val: {}".format(train_loss, val_loss))
        print("MSE: train: {}, val: {}".format(mse_train, mse_val))
        print("KLD: train: {}, val: {}".format(kld_train, kld_val))

        if self.log_data:
            self.visualizer.update_losses(train_loss, val_loss)
            self.visualizer.update_mses(mse_train, mse_val)
            self.visualizer.update_klds(kld_train, kld_val)

            self.saver.update_beta(self.model.beta_updated())
            self.saver.log_csv(train_loss, val_loss, mse_train, mse_val, kld_train, kld_val, val_loss < self.best_loss or self.model.beta_updated())

            if val_loss < self.best_loss or self.model.beta_updated():
                self.saver.save_model(self.model)
                self.best_loss = val_loss

            if epoch % 99 == 0:
                self.visual_trajectories(epoch)
