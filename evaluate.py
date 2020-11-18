from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow import set_random_seed
import glob
import numpy as np
import time
import random
import os

from resnet import ResNet10

from make_images import load_data_from_csv
from make_images import search_all_labels

# Starting to record time.
start = time.time()

############################ User console ##############################
test_file_path = "data/test.csv"
saved_test_file_path = "output/test_pred.csv"

test_ans_file_path = "data/test_ans.csv"
saved_test_ans_file_path = "output/test_ans_pred.csv"
################################# End ##################################

# Set GPU usage.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)


# Numpy arrays path generated from "make_images.py"
test_images_path = "converted_numpy_28_28/test_images.npy"
test_ans_images_path = "converted_numpy_28_28/test_ans_images.npy"

test_labels_path = "converted_numpy_28_28/test_labels.npy"
test_ans_labels_path = "converted_numpy_28_28/test_ans_labels.npy"

# Hyper-parameters.
batch_size = 128
num_classes = 30
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
save_model_name = "output/best_model.hdf5"

# Generate label string and label index mappings.
label_to_index, index_to_label = search_all_labels()
print("======== Label-index mapping ==========")
print("Label index to string mapping: \n", index_to_label)
print()


# Load test data from csv files.
test_images, test_labels = load_data_from_csv(test_file_path)
test_ans_images, test_ans_labels = load_data_from_csv(test_ans_file_path)

# Set data as channel_last format.
test_images = test_images.transpose(0, 2, 3, 1)
test_ans_images = test_ans_images.transpose(0, 2, 3, 1)

print("========== Dataset information ================")
print("test images shape: ", test_images.shape)
print("test_ans images shape: ", test_ans_images.shape)
print()


x_test = test_images.astype('float32')
x_test_ans = test_ans_images.astype('float32')

# Scale image to [0, 1].
x_test /= 255
x_test_ans /= 255

# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(test_labels, num_classes)
y_test_ans = keras.utils.to_categorical(test_ans_labels, num_classes)

# Use the modified Resnet10.
model = ResNet10(input_shape, num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

# Reload best weights and print accuracy.
model.load_weights(save_model_name)

print("========== test accuracy ==============")
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

test_ans_score = model.evaluate(x_test_ans, y_test_ans, verbose=0)
print('Test_ans loss:', test_ans_score[0])
print('Test_ans accuracy:', test_ans_score[1])
print()

# Print and save predictions.
print("============= Test predictions ===============")
test_pred = model.predict(x_test)
test_pred = np.argmax(test_pred, axis=1)
test_pred = test_pred.astype(dtype=np.int32)
test_pred = test_pred.tolist()
test_labels = test_labels.astype(dtype=np.int32)
pred_labels = [index_to_label[pred] for pred in test_pred]
gt_labels = [index_to_label[gt] for gt in test_labels]
print("Test prediction: ", pred_labels)
print("Test label: ", gt_labels)
print()

test_pred_ans = model.predict(x_test_ans)
test_pred_ans = np.argmax(test_pred_ans, axis=1)
test_pred_ans = test_pred_ans.astype(dtype=np.int32)
test_pred_ans = test_pred_ans.tolist()
test_ans_labels = test_ans_labels.astype(dtype=np.int32)
pred_ans_labels = [index_to_label[pred_ans] for pred_ans in test_pred_ans]
gt_ans_labels = [index_to_label[gt_ans] for gt_ans in test_ans_labels]
print("Test_ans prediction: ", pred_ans_labels)
print("Test_ans label: ", gt_ans_labels)
print()

import pandas as pd
# Saving test predictions.
test_pred = pd.DataFrame({"predictions": pred_labels})
test_ans_pred = pd.DataFrame({"predictions": pred_ans_labels})

test_pred.to_csv(saved_test_file_path, index=True, sep=',')
test_ans_pred.to_csv(saved_test_ans_file_path, index=True, sep=',')

# Show total runtime.
end = time.time()
print("Total runtime is {}s".format(end-start))
