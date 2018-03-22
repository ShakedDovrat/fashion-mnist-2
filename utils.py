import os

import matplotlib.pyplot as plt


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_training_history(history, fig_save_file_path):
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

    plt.savefig(fig_save_file_path)

    plt.show()


class PrintModel(object):
    @staticmethod
    def layers_summary(model):
        PrintModel.num_layers(model)
        PrintModel.num_of_real_layers(model)
        PrintModel.num_of_real_layers(model, only_trainable_layers=True)

    @staticmethod
    def line():
        print('_' * 50)

    @staticmethod
    def num_of_real_layers(model, only_trainable_layers=False):

        def _is_real_layer(layer_type):
            return ('Conv' in layer_type) or ('Dense' in layer_type)

        if not only_trainable_layers:
            layers = [l for l in model.layers if _is_real_layer(l.__class__.__name__)]
            print('num of real layers = {}'.format(len(layers)))
        else:
            layers = [l for l in model.layers if (_is_real_layer(l.__class__.__name__) and l.trainable == True)]
            print('num of real trainable layers = {}'.format(len(layers)))

        return len(layers)

    @staticmethod
    def weights_shape(model):
        PrintModel.line()
        PrintModel.line()
        print('{:30s} {:20s}'.format('Layer', 'Weights shape'))
        PrintModel.line()
        for layer in model.layers:
            weights = layer.get_weights()
            if weights != []:
                weights_shape = str(layer.get_weights()[0].shape)
            else:
                weights_shape = 'None'
            print('{:30s} {:20s}'.format(layer.name, weights_shape))
            PrintModel.line()

    @staticmethod
    def num_layers(model):
        print('num of layers = {}'.format(len(model.layers)))
