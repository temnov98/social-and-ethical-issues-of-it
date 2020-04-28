import cv2
import numpy as np
from PIL import Image
import io
from os import listdir
from os.path import isfile, join


# image = read_image(full_path, cv2.COLOR_RGB2BGR)  # COLOR_RGB2GRAY
def read_image(image_path, read_format):
    input_image = open(image_path, "rb").read()
    io_bytes = io.BytesIO(input_image)
    bytes_image = Image.open(io_bytes)
    np_array = np.array(bytes_image)
    return cv2.cvtColor(np_array, read_format)


def get_file_names(directory_path):
    return [f'{directory_path}/{f}' for f in listdir(directory_path) if isfile(join(directory_path, f))]


def read_train_image(image_path):
    try:
        image = read_image(image_path, cv2.COLOR_RGB2GRAY)
    except:
        image = read_image(image_path, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # width = 130
    # height = 130
    #
    # resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    # return resized.flatten()

    return image.flatten()


def get_train_dataset(directory_path, label):
    file_names = get_file_names(directory_path)

    files = [read_train_image(file_name) for file_name in file_names]
    labels = [label for file_name in file_names]

    return [files, labels]


def get_train_data_and_labels(directory_path, labels):
    train_data = []
    train_labels = []

    for label in labels:
        data_set = get_train_dataset(f'{directory_path}/{label}', label)

        train_data = [*train_data, *data_set[0]]
        train_labels = [*train_labels, *data_set[1]]

    return [train_data, train_labels]
