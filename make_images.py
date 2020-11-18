"""This part is for making images from strokes and showing the demo ones."""

import glob
import numpy as np
import os
import pandas
from PIL import Image

import cairocffi as cairo 

############# User console #######################
# Load data from csv files.
demo_file_path = "data/demo.csv"
train_files_path = "data/train/*.csv"
valid_file_path = "data/valid.csv"
test_file_path = "data/test.csv"
test_ans_file_path = "data/test_ans.csv"

size = 28  # 28 for classification, 60 for generation.

############# User console #######################

image_width = size
image_height = size
dump_folder_prefix = "converted_numpy_{}_{}/".format(image_width, image_height)

if not os.path.exists(dump_folder_prefix):
    os.makedirs(dump_folder_prefix)


def vector_to_raster(vector_images, side=size, line_diameter=16, padding=16, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
    """padding and line_diameter are relative to the original 256x256 image.

    Note:
        From https://github.com/googlecreativelab/quickdraw-dataset/issues/19
    """

    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images


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


def convert_to_number(string_labels):
    """Conver text labels to index labels."""
    label_to_index, index_to_label = search_all_labels()
    index_labels = [label_to_index[string_label] for string_label in string_labels]
    return index_labels

def load_data_from_csv(filename):
    """Load data from csv files."""
    df = pandas.read_csv(filename)
    images = df["drawing"].tolist()
    labels = df["word"].tolist()

    images = [eval(image) for image in images]
    index_labels = convert_to_number(labels)

    labels = np.asarray(index_labels, dtype=np.float32)
    raster_images = vector_to_raster(images)
    raster_images = np.asarray(raster_images, dtype=np.float32)

    raster_images = raster_images.reshape((-1, 1, image_width, image_height))
    return raster_images, labels

def convert_train_files(filename):
    """ Search the training dataset to generate label-index mappings."""
    train_files = glob.glob(filename)
    images, labels = [], []
    cnt = 0
    for train_file in train_files:
        image, label = load_data_from_csv(train_file)
        images.extend(image)
        labels.extend(label)
        print(cnt)
        cnt += 1
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    return images, labels


if __name__=="__main__":
    # Show and demo the drawings in "demo.csv".
    show_demo_images, labels = load_data_from_csv(demo_file_path)
    show_demo_images = show_demo_images.reshape((-1, image_width, image_height))
    for idx, image in enumerate(show_demo_images):
        img = Image.fromarray(image)
        img = img.convert("L")
        img.save("demo_images/" + str(idx) +".png")
        img.show()

    # Convert demo.csv to numpy array with 1*image_width*image_height.
    demo_images, demo_labels = load_data_from_csv(demo_file_path)
    print("demo image shape: ", demo_images.shape)  # demo image shape:  (5, 1, image_width, image_height)
    print("demo label shape: ", demo_labels.shape)  # demo label shape:  (5,)
    np.save(dump_folder_prefix + "demo_images.npy", demo_images)
    np.save(dump_folder_prefix + "demo_labels.npy", demo_labels)

    # Convert valid.csv to numpy array with 1*image_width*image_height.
    valid_images, valid_labels = load_data_from_csv(valid_file_path)
    print("valid image shape: ", valid_images.shape)  # valid image shape:  (438086, 1, image_width, image_height)
    print("valid label shape: ", valid_labels.shape)  # valid label shape:  (438086,)
    np.save(dump_folder_prefix + "valid_images.npy", valid_images)
    np.save(dump_folder_prefix + "valid_labels.npy", valid_labels)

    # Convert test.csv to numpy array with 1*image_width*image_height.
    test_images, test_labels = load_data_from_csv(test_file_path)
    print("test image shape: ", test_images.shape)  # test image shape:  (41, 1, image_width, image_height)
    print("test label shape: ", test_labels.shape)  # test label shape:  (41,)
    np.save(dump_folder_prefix + "test_images.npy", test_images)
    np.save(dump_folder_prefix + "test_labels.npy", test_labels)

    # Convert test_ans.csv to numpy array with 1*image_width*image_height.
    test_ans_images, test_ans_labels = load_data_from_csv(test_ans_file_path)
    print("test_ans image shape: ", test_ans_images.shape)  # test image shape:  (41, 1, image_width, image_height)
    print("test_ans label shape: ", test_ans_labels.shape)  # test label shape:  (41,)
    np.save(dump_folder_prefix + "test_ans_images.npy", test_ans_images)
    np.save(dump_folder_prefix + "test_ans_labels.npy", test_ans_labels)

    # Convert train/*.csv to numpy array with 1*image_width*image_height.
    train_images, train_labels = convert_train_files(train_files_path)
    print("train image shape: ", train_images.shape)  # train image shape:  (3943968, 1, image_width, image_height)
    print("train label shape: ", train_labels.shape)  # train label shape:  (3943968,)
    np.save(dump_folder_prefix + "train_images.npy", train_images)
    np.save(dump_folder_prefix + "train_labels.npy", train_labels)
