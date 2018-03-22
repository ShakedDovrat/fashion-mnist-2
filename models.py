from keras.layers import Input, Dense, Activation, Flatten, Conv2D, BatchNormalization, Dropout, \
    GlobalAveragePooling2D, MaxPool2D
from keras.models import Model, Sequential

from DenseNet import densenet


def shallow_model(input_shape):
    img_input = Input(input_shape)

    x = Conv2D(16, (3, 3), padding='valid')(img_input)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)


def deep_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    # model.add(MaxPool2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    # model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def dense_net_bc_wide_40(input_shape):
    base = densenet.DenseNet(input_shape=input_shape, depth=40, nb_dense_block=3, growth_rate=24,
                             bottleneck=True, reduction=0.5, dropout_rate=0.0,
                             weight_decay=1e-4, subsample_initial_block=False, include_top=False, classes=10)
    x = Dense(10, activation='softmax')(base.layers[-1].output)
    return Model(inputs=base.input, outputs=x)


def dense_net_bc_40(input_shape):
    base = densenet.DenseNet(input_shape=input_shape, depth=40, nb_dense_block=3, growth_rate=12,
                             bottleneck=True, reduction=0.5, dropout_rate=0.0,
                             weight_decay=1e-4, subsample_initial_block=False, include_top=False, classes=10)
    x = Dense(10, activation='softmax')(base.layers[-1].output)
    return Model(inputs=base.input, outputs=x)
