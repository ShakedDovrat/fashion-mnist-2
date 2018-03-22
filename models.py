from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.models import Model


def shallow_model(image_size):
    img_input = Input(image_size)

    x = Conv2D(16, (3, 3), padding='valid')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)

