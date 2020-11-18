from __future__ import division

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


############### BN, conv, relu block ###############33
def _bn_relu(x, bn_name=None, relu_name=None):
    norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)

def _conv_bn_relu(**block_config):
    filters = block_config["filters"]
    kernel_size = block_config["kernel_size"]
    strides = block_config.setdefault("strides", (1, 1))
    dilation_rate = block_config.setdefault("dilation_rate", (1, 1))
    conv_name = block_config.setdefault("conv_name", None)
    bn_name = block_config.setdefault("bn_name", None)
    relu_name = block_config.setdefault("relu_name", None)
    kernel_initializer = block_config.setdefault("kernel_initializer", "he_normal")
    padding = block_config.setdefault("padding", "same")
    kernel_regularizer = block_config.setdefault("kernel_regularizer", l2(1.e-4))
    def block_func(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name)

    return block_func

def _bn_relu_conv(**block_config):
    filters = block_config["filters"]
    kernel_size = block_config["kernel_size"]
    strides = block_config.setdefault("strides", (1, 1))
    dilation_rate = block_config.setdefault("dilation_rate", (1, 1))
    conv_name = block_config.setdefault("conv_name", None)
    bn_name = block_config.setdefault("bn_name", None)
    relu_name = block_config.setdefault("relu_name", None)
    kernel_initializer = block_config.setdefault("kernel_initializer", "he_normal")
    padding = block_config.setdefault("padding", "same")
    kernel_regularizer = block_config.setdefault("kernel_regularizer", l2(1.e-4))
    def block_func(x):
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)
    return block_func

############ Short-cutting for resnet skip-connection. #############
def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = BatchNormalization(axis=CHANNEL_AXIS,
                                      name=bn_name_base)(shortcut)

    return add([shortcut, residual])

############# Residual block for resnet. #################
def _residual_block(block_function, filters, blocks, stage,
                    transition_strides=None, transition_dilation_rates=None,
                    dilation_rates=None, is_first_layer=False, dropout=None,
                    residual_unit=_bn_relu_conv):
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks
    def block_func(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=is_first_block,
                               dropout=dropout,
                               residual_unit=residual_unit)(x)
        return x
    return block_func

def _block_name_base(stage, block):
    """Helper function for block name."""
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base

def basic_block(filters, stage, block, transition_strides=(1, 1),
                dilation_rate=(1, 1), is_first_block_of_first_layer=False,
                dropout=None, residual_unit=_bn_relu_conv):
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            x = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)

        if dropout is not None:
            x = Dropout(dropout)(x)
        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        return _shortcut(input_features, x)
    return f

def ResNet(input_shape, classes, block, repetitions,
           initial_filters=64, activation='softmax',
           input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
           initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max'):
    block_fn = block
    residual_unit = _bn_relu_conv
    img_input = Input(shape=input_shape, tensor=input_tensor)
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                      strides=1)(img_input)
    if initial_pooling == 'max':
        x = MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(x)
    block = x
    filters = initial_filters
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        block = _residual_block(block_fn, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit)(block)
        filters *= 2
    x = _bn_relu(block)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=classes, activation=activation,
              kernel_initializer="he_normal")(x)
    model = Model(inputs=img_input, outputs=x)
    return model

def ResNet10(input_shape, classes):
    return ResNet(input_shape, classes, basic_block, repetitions=[2, 2])
