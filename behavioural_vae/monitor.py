import numpy as np
import torch

import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import MeterLogger
from tqdm import tqdm
import csv
import os


class Trainer(Engine):

    def __init__(self, dataloader, model, latent_dim, save_folder=None, save_name=None, log=False, visdom=True, server='localhost', port=8097, visdom_title="mnist_meterlogger", debug=False):
        super(Trainer, self).__init__()

        self.debug = debug
        self.get_iterator = dataloader.get_iterator
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.initialize_engine()

        self.model = model
        self.log_data = log

        if self.log_data:
            assert(save_folder is not None and save_name is not None)
            self.initilize_log(save_folder, save_name)
            self.best_loss = np.inf
        else:
            assert(save_folder is None and save_name is None)
        self.visdom = visdom
        self.mlog = MeterLogger(server=server, port=port, title=visdom_title)

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
        if self.visdom:
            self.mlog.update_loss(loss, meter='loss')

    def on_start_epoch(self, state):

        self.model.train(True)
        self.reset_meters()
        if self.visdom:
            self.mlog.timer.reset()

        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state):

        train_loss = self.meter_loss.value()[0]
        self.reset_meters()
        if self.visdom:

            self.mlog.print_meter(mode="Train", iepoch=state['epoch'])
            self.mlog.reset_meter(mode="Train", iepoch=state['epoch'])

            self.test(self.model.evaluate, self.get_iterator(False))
            val_loss = self.meter_loss.value()[0]

            if self.log_data:

                self.log_csv(train_loss, val_loss, val_loss < self.best_loss)

                if val_loss < self.best_loss:
                    self.save_model()
                    self.best_loss = val_loss
            self.mlog.print_meter(mode="Test", iepoch=state['epoch'])
            self.mlog.reset_meter(mode="Test", iepoch=state['epoch'])

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


#class Demonstrator(Trainer):
#
#    def __init__(self,  folder, model_name, model, data_loader, visdom_title='training_results'):
#        super(Demonstrator, self).__init__(data_loader, visdom_title=visdom_title, visdom=True)
#        self.model = model
#        self.load_parameters(folder, model_name)
#
#    def initialize_engine(self):
#        self.hooks['on_sample'] = self.on_sample
#        self.hooks['on_forward'] = self.on_forward
#
#    def load_parameters(self, folder, model_name):
#        Path = os.path.join('log/{}'.format(folder), '{}.pth.tar'.format(model_name))
#        self.model.load_state_dict(torch.load(Path))
#        self.model.eval()
#
#    def evaluate(self):
#        self.test(self.model.evaluate, self.get_iterator(False))
#        val_loss = self.meter_loss.value()[0]
#
#        print('Testing loss: %.4f' % (val_loss))
#
#        self.generate_visdom_samples(self.visdom_samples)
#        self.generate_latent_samples(self.visdom_samples[0])



