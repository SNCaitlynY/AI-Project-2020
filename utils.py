import math
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set matplot canvas as large as containing the generated image grid.
matplotlib.use('Agg')
plt.rcParams['savefig.dpi'] = 400  
plt.rcParams['figure.dpi'] = 400


def combine_images(generated_images):
    """Arrange generated_images as grid_image (30 * 30)."""
    num_images = generated_images.shape[0]
    new_width = int(math.sqrt(num_images))
    new_height = int(math.ceil(float(num_images) / new_width))
    grid_shape = generated_images.shape[1:3]
    grid_image = np.zeros((new_height * grid_shape[0], new_width * grid_shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / new_width)
        j = index % new_width
        grid_image[i * grid_shape[0]:(i + 1) * grid_shape[0], j * grid_shape[1]:(j + 1) * grid_shape[1]] = \
            img[:, :, 0]
    return grid_image


def generate_noise(shape):
    """Generate noise as latent vector for generator input."""
    noise = np.random.uniform(0, 1, size=shape)
    return noise


def generate_condition_embedding(label, nb_of_label_embeddings):
    """Generate a label-conditional embeddings for generator."""
    label_embeddings = np.zeros((nb_of_label_embeddings, 100))
    label_embeddings[:, label] = 1
    return label_embeddings


def generate_images(generator, nb_images, label):
    """From noist to generated images."""
    noise = generate_noise((nb_images, 100))
    label_batch = generate_condition_embedding(label, nb_images)
    generated_images = generator.predict([noise, label_batch], verbose=0)
    return generated_images


def generate_image_grid(generator, title="Generated images"):
    """Generate image grid from images."""
    generated_images = []
    # Here we generated 30*30 grid for visualization.
    for i in range(30):
        noise = generate_noise((30, 100))
        label_input = generate_condition_embedding(i, 30)
        gen_images = generator.predict([noise, label_input], verbose=0)
        generated_images.extend(gen_images)
    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    image_grid = inverse_transform_images(image_grid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image_grid, cmap="gray")
    ax.set_title(title)
    fig.canvas.draw()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def save_generated_image(image, prefix, suffix, folder_path):
    """Save the generated image."""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    file_path = "{}/{}_{}.png".format(folder_path, prefix, suffix)
    cv2.imwrite(file_path, image.astype(np.uint8))


def transform_images(images):
    """Transform images to [-1, 1] to match with tanh activation function."""
    images = (images.astype(np.float32) - 127.5) / 127.5
    return images


def inverse_transform_images(images):
    """Tranform the images with [-1, 1] to [0, 255]."""
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images


def save_csv_image(image, prefix, suffix, folder_path):
    """Save the generated images to csv."""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    file_path = "{}/{}_{}.csv".format(folder_path, prefix, suffix)
    np.savetxt(file_path, image, delimiter = ',')

