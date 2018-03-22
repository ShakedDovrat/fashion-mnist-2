import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist

from utils import ensure_dir
from models import *


class Config(object):
    def __init__(self):
        self.image_size = (28, 28, 1)
        self.model_func = shallow_model
        self.model_name = self.model_func.__name__
        self.lr = 1e-4
        self.batch_size = 32
        self.epochs = 5


class Model(object):
    def __init__(self):
        self.logs_dir = 'logs'
        ensure_dir(self.logs_dir)
        self.c = Config()
        self.run_name = Model._get_run_name()
        self.model = None
        self.train_set = None
        self.test_set = None

    def run(self):
        self.model = self._build_model()
        self._compile_model()
        self.train_set, self.test_set = Model._load_data()
        history = self.train()
        self._plot_training_history(history)
        self.test()

    def train(self):
        callbacks = self._get_callbacks()
        history = self.model.fit(self.train_set['x'],
                                 self.train_set['y'],
                                 batch_size=self.c.batch_size,
                                 epochs=self.c.epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_split=0.2)
        return history

    def test(self):
        metrics_list = self.model.evaluate(self.test_set['x'], self.test_set['y'], verbose=1)
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
        return model

    def _compile_model(self):
        loss = 'sparse_categorical_crossentropy'  # 'categorical_crossentropy'
        optimizer = Adam(lr=self.c.lr)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _get_callbacks(self):
        checkpoint_writer = ModelCheckpoint('{}-weights.h5'.format(self.run_name),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')

        return [checkpoint_writer]

    @staticmethod
    def _load_data():
        trainset, testset = fashion_mnist.load_data()
        return Model._reformat_dataset(trainset), Model._reformat_dataset(testset)

    @staticmethod
    def _reformat_dataset(dataset):
        x, y = dataset
        x = np.reshape(x, list(x.shape) + [1])
        return {'x': x, 'y': y}

    def _plot_training_history(self, history):
        plt.figure(1)

        # summarize history for accuracy

        plt.subplot(211)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        # summarize history for loss

        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.savefig('logs/{}.png'.format(self.run_name))

        plt.show()


# def dummy_model(image_size):
#     return 0


def main():
    run = Model()
    run.run()


main()
