import os
import numpy as np
from keras import backend as K
from keras import utils as keras_utils
from keras import optimizers
from keras import datasets
import utils
import random
from tensorflow import set_random_seed

import models
from make_images import search_all_labels

############# User console ################
generator_model_path = "models/generator.h5"
saved_image_prefix = "output/image"
saved_csv_prefix = "output/csv"
############# User console ################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

# Label index mapping.
label_to_index, index_to_label = search_all_labels()

# Construct generator model and then load trained model weights.
print("Generator:")
G = models.generator_model()
G.load_weights(generator_model_path)
G.summary()

# Generate images conditionally.
for i in range(30):
    label = index_to_label[i]  # Get the labels.
    images = utils.generate_images(G, nb_images=10, label=i)  # Generate images.
    images = utils.inverse_transform_images(images)  # Scale images from [-1, 1] to [0, 255].
    for j in range(10):
        utils.save_generated_image(images[j], label, j, saved_image_prefix)  # Save png files.
        utils.save_csv_image(images[j, :, :, 0].astype(dtype=np.int32), label, j, saved_csv_prefix)  # Save 2-d array into csv files.
