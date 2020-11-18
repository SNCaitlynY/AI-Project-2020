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

# Starting to record time.
start = time.time()

############################ User console ##############################
# Set training or testing.
training = True
read_test_from_csv = True  # True for reading from csv, false for loading dumped numpy arrays.

test_file_path = "data/test.csv"
saved_test_file_path = "output/test_pred.csv"

test_ans_file_path = "data/test_ans.csv"
saved_test_ans_file_path = "output/test_ans_pred.csv"
################################# End ##################################

# Below are the dumped numpy arrays.
train_images_path = "converted_numpy_28_28/train_images.npy"
valid_images_path = "converted_numpy_28_28/valid_images.npy"
train_labels_path = "converted_numpy_28_28/train_labels.npy"
valid_labels_path = "converted_numpy_28_28/valid_labels.npy"

# Set GPU usage.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

# Training and test file paths.
train_files_path = "data/train/*.csv"

# Numpy arrays path generated from "make_images.py"
demo_images_path = "converted_numpy_28_28/demo_images.npy"
test_images_path = "converted_numpy_28_28/test_images.npy"
test_ans_images_path = "converted_numpy_28_28/test_ans_images.npy"

demo_labels_path = "converted_numpy_28_28/demo_labels.npy"
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


def search_all_labels():
    """ Search the training dataset to generate label-index mappings."""
    label_to_index = {}
    index_to_label = {}
    train_files = glob.glob(train_files_path)
    basenames = [os.path.basename(train_file) for train_file in train_files]
    index = 0
    for basename in basenames:
        name, ext = os.path.splitext(basename)
        if ext == ".csv":
            label_to_index[name] = index
            index_to_label[index] = name
            index += 1
    return label_to_index, index_to_label


def unison_shuffled_copies(images, labels):
    """Helper function for image-label random shuffle."""
    assert len(images) == len(labels)
    p = np.random.permutation(len(images))
    return images[p], labels[p]


# Generate label string and label index mappings.
label_to_index, index_to_label = search_all_labels()
print("======== Label-index mapping ==========")
print("Label index to string mapping: \n", index_to_label)
print()

# Load training, validation and test dataset from the dumped numpy arrays.
train_images = np.load(train_images_path)
train_labels = np.load(train_labels_path)

valid_images = np.load(valid_images_path)
valid_labels = np.load(valid_labels_path)

if read_test_from_csv:
    # Load test data from csv files.
    test_images, test_labels = load_data_from_csv(test_file_path)
    test_ans_images, test_ans_labels = load_data_from_csv(test_ans_file_path)
else:
    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path)

    test_ans_images = np.load(test_ans_images_path)
    test_ans_labels = np.load(test_ans_labels_path)

# Set data as channel_last format.
train_images = train_images.transpose(0, 2, 3, 1)
valid_images = valid_images.transpose(0, 2, 3, 1)
test_images = test_images.transpose(0, 2, 3, 1)
test_ans_images = test_ans_images.transpose(0, 2, 3, 1)

# Shuffle training data.
train_images, train_labels = unison_shuffled_copies(train_images, train_labels)

print("========== Dataset information ================")
print("train images shape: ", train_images.shape)
print("valid images shape: ", valid_images.shape)
print("test images shape: ", test_images.shape)
print("test_ans images shape: ", test_ans_images.shape)
print()


x_train = train_images.astype('float32')
x_valid = valid_images.astype('float32')
x_test = test_images.astype('float32')
x_test_ans = test_ans_images.astype('float32')

# Training data augmentation.
# 1. Rescale to [0, 1]
# 2. random shear ratio: 0.2
# 3. random zoom ratio: 0.2
# 4. horizontal flip: enable
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

# Scale image to [0, 1].
x_valid /= 255
x_test /= 255
x_test_ans /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(train_labels, num_classes)
y_valid = keras.utils.to_categorical(valid_labels, num_classes)
y_test = keras.utils.to_categorical(test_labels, num_classes)
y_test_ans = keras.utils.to_categorical(test_ans_labels, num_classes)

# Use the modified Resnet10.
model = ResNet10(input_shape, num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

# Make training data generator.
train_datagen.fit(x_train)

if training:
    # Training.
    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[
            ModelCheckpoint(save_model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ]
    )

# Reload best weights and print accuracy.
model.load_weights(save_model_name)

print("========== validation and test accuracy ==============")
valid_score = model.evaluate(x_valid, y_valid, verbose=0)
print('Valid loss:', valid_score[0])
print('Valid accuracy:', valid_score[1])

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
