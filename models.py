from keras.layers import Input, Dense, Activation, Flatten, Conv2D, BatchNormalization, Dropout, \
    GlobalAveragePooling2D, MaxPool2D
from keras.models import Model, Sequential


def shallow_model(image_size):
    img_input = Input(image_size)

    x = Conv2D(16, (3, 3), padding='valid')(img_input)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)


def deep_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=image_size))
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
