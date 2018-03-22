import datetime
import os

import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator

from utils import ensure_dir, plot_training_history, PrintModel
from models import *
from cutout_random_erasing.random_eraser import get_random_eraser


class Config(object):
    def __init__(self):
        self.image_size = (28, 28, 1)
        self.model_func = dense_net_bc_wide_40  # dense_net_bc_40
        self.model_name = self.model_func.__name__
        self.lr = 1e-2
        self.batch_size = 256
        self.epochs = 100


class Model(object):
    def __init__(self):
        self._logs_dir = 'logs'
        ensure_dir(self._logs_dir)
        self.c = Config()
        self._run_name = Model._get_run_name()
        self.model = None  # TODO: self._model?
        self._datasets = {}
        self._data_generators = {}

    def run(self):
        self.model = self._build_model()
        self._compile_model()
        self._load_data()
        self._create_data_generators()
        history = self.train()
        plot_training_history(history, os.path.join(self._logs_dir, '{}.png'.format(self._run_name)))
        # self.test()

    def train(self):
        callbacks = self._get_callbacks()
        history = self.model.fit_generator(self._data_generators['train'],
                                           epochs=self.c.epochs,
                                           verbose=1,
                                           callbacks=callbacks,
                                           validation_data=self._data_generators['val'],
                                           shuffle=True)

        return history

    def test(self):
        metrics_list = self.model.evaluate(self._datasets['test']['x'], self._datasets['test']['y'], verbose=1)
        print('Test results:')
        for name, value in zip(self.model.metrics_names, metrics_list):
            print('{} = {}'.format(name, value))

    @staticmethod
    def _get_run_name():
        return datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

    def _build_model(self):
        model = self.c.model_func(self.c.image_size)
        print('Using {}:'.format(self.c.model_name))
        model.summary()
        PrintModel.layers_summary(model)
        return model

    def _compile_model(self):
        loss = 'sparse_categorical_crossentropy'  # 'categorical_crossentropy'
        # optimizer = Adam(lr=self.c.lr)
        optimizer = SGD(lr=self.c.lr, momentum=0.9, nesterov=True)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _get_callbacks(self):
        checkpoint_writer = ModelCheckpoint(os.path.join(self._logs_dir, '{}-weights.h5'.format(self._run_name)),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, verbose=1, epsilon=1e-3)
        early_stop = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=15, verbose=1)
        return [checkpoint_writer, reduce_lr, early_stop]

    def _load_data(self):
        trainset, testset = fashion_mnist.load_data()
        train_and_val = Model._reformat_dataset(trainset)
        self._datasets['test'] = Model._reformat_dataset(testset)
        self._datasets['train'], self._datasets['val'] = self._split_train_val(train_and_val)

    @staticmethod
    def _reformat_dataset(dataset):
        x, y = dataset
        x = np.reshape(x, list(x.shape) + [1])
        x = x / 255.0
        # x = x - 127
        return {'x': x, 'y': y}

    def _split_train_val(self, train_and_val):
        num_samples = len(train_and_val['x'])
        val_size = len(self._datasets['test']['x'])
        val_idxs = np.random.choice(num_samples, size=val_size, replace=False)
        train_idxs = list(set(range(num_samples)) - set(val_idxs))
        train = {'x': train_and_val['x'][train_idxs],
                 'y': train_and_val['y'][train_idxs]}
        val = {'x': train_and_val['x'][val_idxs],
               'y': train_and_val['y'][val_idxs]}
        return train, val

    def _create_data_generators(self):
        # augmentations = {'preprocessing_function': get_random_eraser(v_l=0.0, v_h=1.0),
        # augmentations = {'rotation_range': 8,
        #                  'width_shift_range': 0.08,
        #                  'height_shift_range': 0.08,
        #                  'shear_range': 0.3,
        #                  'zoom_range': 0.08,
        #                  'horizontal_flip': True}
        augmentations = {'rotation_range': 4,
                         'width_shift_range': 0.08,
                         'height_shift_range': 0.08,
                         'shear_range': 0.1,
                         'zoom_range': 0.08}
        for dataset in ('train', 'val', 'test'):
            curr_aug = augmentations if dataset == 'train' else {}
            generator = ImageDataGenerator(**curr_aug)
            # zca_whitening=True,
            # featurewise_center=True)#False)
            # featurewise_std_normalization=False)#True)
            generator.fit(self._datasets['train']['x'])  # fit according to train set.
            self._data_generators[dataset] = generator.flow(self._datasets[dataset]['x'],
                                                            self._datasets[dataset]['y'],
                                                            batch_size=self.c.batch_size)


def main():
    run = Model()
    run.run()


main()
