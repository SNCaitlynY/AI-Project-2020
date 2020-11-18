import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from keras import backend as K
from keras import utils as keras_utils
from keras import optimizers
from keras import datasets
from tqdm import tqdm
import models 
import utils

import random
from tensorflow import set_random_seed

from make_images import convert_train_files

# Set matplot canvas as large as containing the generated image grid.
matplotlib.use('Agg')
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 400

################# User console ##########################
BATCH_SIZE = 512
EPOCHS = 1  # Only one epoch is good enough for generation.
read_train_from_csv = False  # True for reading from csv, false for loading dumped numpy arrays.

train_files_path = "data/train/*.csv"
train_images_path = "converted_numpy_60_60/train_images.npy"
train_labels_path = "converted_numpy_60_60/train_labels.npy"
################# User console ##########################

# Set random seed
SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if read_train_from_csv:
    # Load training data from csv files.
    train_images, train_labels = convert_train_files(train_files_path)
else:
    # Reading from the dumped numpy files.
    train_images = np.load(train_images_path)
    train_labels = np.load(train_labels_path)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

train_images, train_labels = unison_shuffled_copies(train_images, train_labels)
train_images = train_images.transpose(0, 2, 3, 1)  # Set channel_last format.
train_images = train_images.reshape(-1, 60, 60)

print("========== Dataset information ================")
print("Train images shape: ", train_images.shape)
print()

# Transform data format.
X_train = train_images
y_train = train_labels
X_train = utils.transform_images(X_train)  # Normalize to [-1, 1]
X_train = X_train[:, :, :, None]
y_train = keras_utils.to_categorical(y_train, 100)  # One hot encoding.

# Create the models
print("Generator:")
G = models.generator_model()
G.summary()

print("Discriminator:")
D = models.discriminator_model()
D.summary()

print("Combined:")
GD = models.generator_containing_discriminator(G, D)
GD.summary()

# Set optimizer and compile models.
optimizer = optimizers.Adam(0.0002, 0.5)
G.compile(loss='binary_crossentropy', optimizer=optimizer)
GD.compile(loss='binary_crossentropy', optimizer=optimizer)
D.trainable = True
D.compile(loss='binary_crossentropy', optimizer=optimizer)

iteration = 0
iter_per_epoch = int(X_train.shape[0] / BATCH_SIZE)
for epoch in range(EPOCHS):
    pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])

    g_losses_for_epoch = []
    d_losses_for_epoch = []

    for i in range(iter_per_epoch):
        noise = utils.generate_noise((BATCH_SIZE, 100))

        image_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        label_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        # Trainig discriminator.
        generated_images = G.predict([noise, label_batch], verbose=0)
        if i % 20 == 0:
            image_grid = utils.generate_image_grid(G, title="Epoch {0}, iteration {1}".format(epoch, iteration))
            utils.save_generated_image(image_grid, epoch, i, "images/iterations")
        X = np.concatenate((image_batch, generated_images))
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        label_batches_for_discriminator = np.concatenate((label_batch, label_batch))
        D_loss = D.train_on_batch([X, label_batches_for_discriminator], y)
        d_losses_for_epoch.append(D_loss)

        # Training generaotr.
        noise = utils.generate_noise((BATCH_SIZE, 100))
        D.trainable = False
        G_loss = GD.train_on_batch([noise, label_batch], [1] * BATCH_SIZE)
        D.trainable = True
        g_losses_for_epoch.append(G_loss)

        pbar.update(BATCH_SIZE)
        iteration += 1

    image_grid = utils.generate_image_grid(G, title="Epoch {0}".format(epoch))
    utils.save_generated_image(image_grid, epoch, 0, "images/epochs")
    pbar.close()
    print("D loss: {0}, G loss: {1}".format(np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)))

    G.save_weights("models/generator.h5")
    D.save_weights("models/discriminator.h5")
