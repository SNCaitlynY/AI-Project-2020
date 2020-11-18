from keras import models
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Reshape, Concatenate, UpSampling2D
from keras.layers import MaxPooling2D, Conv2D, Flatten


def generator_model():
    """Conditional generator."""
    # Latent vector input.
    input_z = Input((100,))
    dense_z_1 = Dense(1024)(input_z)  # Fully conected to 1024.
    act_z_1 = Activation("tanh")(dense_z_1)
    dense_z_2 = Dense(128 * 15 * 15)(act_z_1)  # Fully connected to 128 * (60/4) * (60/4)
    bn_z_1 = BatchNormalization()(dense_z_2)
    reshape_z = Reshape((15, 15, 128), input_shape=(128 * 15 * 15,))(bn_z_1)  # Reshape as a square.

    # Conditional labels input.
    input_c = Input((100,))
    dense_c_1 = Dense(1024)(input_c)  # Fully conected to 1024.
    act_c_1 = Activation("tanh")(dense_c_1)
    dense_c_2 = Dense(128 * 15 * 15)(act_c_1)  # Fully connected to 128 * (60/4) * (60/4)
    bn_c_1 = BatchNormalization()(dense_c_2)
    reshape_c = Reshape((15, 15, 128), input_shape=(128 * 15 * 15,))(bn_c_1)  # Reshape as a square.

    # Combine latent vector and the label input as generator input.
    concat_z_c = Concatenate()([reshape_z, reshape_c])

    # Using upsampling and activation to generate images conditionally.
    up_1 = UpSampling2D(size=(2, 2))(concat_z_c)
    conv_1 = Conv2D(64, (5, 5), padding='same')(up_1)
    act_1 = Activation("tanh")(conv_1)
    up_2 = UpSampling2D(size=(2, 2))(act_1)
    conv_2 = Conv2D(1, (5, 5), padding='same')(up_2)
    act_2 = Activation("tanh")(conv_2)
    model = models.Model(inputs=[input_z, input_c], outputs=act_2)
    return model


def discriminator_model():
    """Conditional discriminator."""
    # Input image 60 * 60 * 1.
    input_gen_image = Input((60, 60, 1))

    # Conv-tanh-maxpooling.
    conv_1_image = Conv2D(64, (5, 5), padding='same')(input_gen_image)
    act_1_image = Activation("tanh")(conv_1_image)
    pool_1_image = MaxPooling2D(pool_size=(2, 2))(act_1_image)

    # Conv-tanh-maxpooling.
    conv_2_image = Conv2D(128, (5, 5))(pool_1_image)
    act_2_image = Activation("tanh")(conv_2_image)
    pool_2_image = MaxPooling2D(pool_size=(2, 2))(act_2_image)

    # Input labels.
    input_c = Input((100,))
    dense_1_c = Dense(1024)(input_c)
    act_1_c = Activation("tanh")(dense_1_c)
    dense_2_c = Dense(13 * 13 * 128)(act_1_c)  # Fully connected to 128 * (60/4-2) * (60/4-2)
    bn_c = BatchNormalization()(dense_2_c)
    reshaped_c = Reshape((13, 13, 128))(bn_c)  # Reshape as a square.
    
    # Concate the image features and label features.
    concat = Concatenate()([pool_2_image, reshaped_c])

    # Discriminator.
    flat = Flatten()(concat)
    dense_1 = Dense(1024)(flat)
    act_1 = Activation("tanh")(dense_1)
    dense_2 = Dense(1)(act_1)
    act_2 = Activation('sigmoid')(dense_2)
    model = models.Model(inputs=[input_gen_image, input_c], outputs=act_2)
    return model


def generator_containing_discriminator(g, d):
    """Stack generator and discriminator as the whole model."""
    input_z = Input((100,))  # latent vector.
    input_c = Input((100,))  # input labels.
    gen_image = g([input_z, input_c])  # Generate image conditionally.
    d.trainable = False  # Set discriminator non-trainable.
    is_real = d([gen_image, input_c])
    model = models.Model(inputs=[input_z, input_c], outputs=is_real)
    return model
