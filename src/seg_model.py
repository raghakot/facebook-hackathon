from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K


IMG_WIDTH = 64
IMG_HEIGHT = 64


def _conv_bn_elu(nb_filter, kernel_size, strides=(1, 1)):
    def f(input):
        conv = Conv2D(nb_filter, kernel_size, strides=strides,
                      kernel_initializer='he_normal', padding='same')(input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)
    return f


def build_model1(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """Override input shape to run on larger image.
    """

    inp = Input(shape=input_shape)

    x = _conv_bn_elu(16, 3)(inp)
    x = _conv_bn_elu(16, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(48, 3)(x)
    x = _conv_bn_elu(48, 3)(x)
    x = MaxPooling2D()(x)

    # This kernel size should change in IMG_HEIGHT, IMG_WIDTH changes to output (1, 1, 1) activation map
    x = Conv2D(1, (4, 4), kernel_initializer='he_normal', activation='sigmoid')(x)
    x = Flatten()(x)
    return Model(inp, x)


if __name__ == '__main__':
    model = build_model1()
    model.summary()
