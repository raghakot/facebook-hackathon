from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import backend as K


from seg_model import build_model1 as build_seg_model


IMG_WIDTH = 320
IMG_HEIGHT = 180


def _conv_bn_elu(nb_filter, kernel_size, strides=(1, 1)):
    def f(input):
        conv = Conv2D(nb_filter, kernel_size, strides=strides,
                      kernel_initializer='he_normal', padding='same')(input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)
    return f


def build_model1():
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

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

    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)
    x = Lambda(lambda x: K.sum(x, axis=[1, 2]))(x)
    return Model(inp, x)


def build_model2():
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

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

    # Mask detection
    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)

    # Mask refinement
    x = _conv_bn_elu(4, 3)(x)
    x = _conv_bn_elu(4, 3)(x)
    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)
    x = Lambda(lambda x: K.sum(x, axis=[1, 2]))(x)
    return Model(inp, x)


def build_with_seg_model(seg_model_path):
    seg_model = build_seg_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    seg_model.load_weights(seg_model_path, by_name=True)

    # Create new model without flatten layer.
    main_model = Model(inputs=seg_model.input, outputs=seg_model.layers[-2].output)
    return main_model

if __name__ == '__main__':
    model = build_with_seg_model('../models/seg_2017-10-07-12-03-33.hdf5')
