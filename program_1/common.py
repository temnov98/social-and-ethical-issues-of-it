import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from PIL import Image
import io
import random


default_dimension = (200, 200)  # (width, height)


def read_image(image_path, read_format):
    input_image = open(image_path, "rb").read()
    io_bytes = io.BytesIO(input_image)
    bytes_image = Image.open(io_bytes)
    np_array = np.array(bytes_image)
    return cv2.cvtColor(np_array, read_format)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def save_to_file(image, filename):
    im = Image.fromarray(image)
    im.save(filename)


def adjust_image(x, a, b, c, d, gamma=1):
    return (((x - a) / (b - a)) ** gamma) * (d - c) + c


def step_image(image, threshold):
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]


def power_image(image, scale_multiplier, gamma):
    return scale_multiplier * image ** gamma


def inverse_image(image):
    return 255 - image % 255


def thresh_image(image, threshold):
    return cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)[1]


def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]


def crop_with_saving_size(image, angle, delta):
    width = image.shape[0]
    height = image.shape[1]

    rotated = rotate_image(image, angle)
    cropped = crop_image(rotated, delta, delta, width - delta * 2, height - delta * 2)
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def get_images_with_filters(source_image):
    laplacian = cv2.Laplacian(source_image, cv2.CV_8U)
    image_with_laplacian = source_image + laplacian

    sobel_image = cv2.Sobel(source_image, cv2.CV_64F, 0, 1, ksize=5)

    sobel_kernel = np.ones((5, 5), np.float32) / 25
    image_with_sobel_filter = cv2.filter2D(sobel_image, -1, sobel_kernel)

    multiplication_images = image_with_laplacian * image_with_sobel_filter
    sum_images = source_image + multiplication_images

    sum_images_arr = np.asarray(sum_images)
    gradient_image = adjust_image(sum_images_arr, sum_images_arr.min(), sum_images_arr.max(), 0, 1)

    stepped = step_image(source_image, 150)
    powered = power_image(source_image, 2, 1)

    inversed = inverse_image(source_image)
    threshed = thresh_image(source_image, 150)

    gaussian_blur_1 = cv2.GaussianBlur(source_image, (5, 5), 0)
    bilateral_filter_1 = cv2.bilateralFilter(source_image, 9, 75, 75)
    blur_1 = cv2.blur(source_image, (4, 4))
    gaussian_blur_2 = cv2.GaussianBlur(source_image, (5, 5), cv2.BORDER_ISOLATED)
    bilateral_filter_2 = cv2.bilateralFilter(source_image, 15, 100, 100)

    return [
        source_image,
        image_with_laplacian,
        # gradient_image, # TODO: error when saving to file
        # stepped, # слишком темное получается
        powered,
        # inversed,
        # threshed, # слишком темное получается
        gaussian_blur_1,
        bilateral_filter_1,
        blur_1,
        gaussian_blur_2,
        bilateral_filter_2,
    ]

    # cv2.imshow('Laplacian', image_with_laplacian)
    # cv2.imshow('Gradient', gradient_image)
    # cv2.imshow('Step', stepped)
    # cv2.imshow('Power', powered)
    # cv2.imshow('inversed', inversed)
    # cv2.imshow('threshed', threshed)
    # cv2.imshow('Gauss filter 1', gaussian_blur_1)
    # cv2.imshow('Batterwort filter 1', bilateral_filter_1)
    # cv2.imshow('Ideal filter 1', blur_1)
    # cv2.imshow('Gauss filter 2', gaussian_blur_2)
    # cv2.imshow('Batterwort filter 2', bilateral_filter_2)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_rotated_images(image):
    img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    img_rotate_180_clockwise = cv2.rotate(img_rotate_90_clockwise, cv2.ROTATE_90_CLOCKWISE)
    img_rotate_90_counter_clockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return [
        image,
        img_rotate_90_clockwise,
        img_rotate_180_clockwise,
        img_rotate_90_counter_clockwise,
    ]


def get_cropped_and_rotated_images(image):
    return [image, crop_with_saving_size(image, 10, 15), crop_with_saving_size(image, -10, 15)]


def rand(start_number, stop_number):
    return random.randrange(start_number, stop_number, 1)


def draw_random_circle(image):
    width, height = image.shape
    return cv2.circle(image, (rand(0, width), rand(0, height)), rand(0, 20), (50, 50, 0), -1)


def get_images_with_circles(image):
    return [image, draw_random_circle(image.copy())]


def flip_horizontal(image):
    # отразим изображение: 0 – по вертикали, 1 – по горизонтали, (-1) – по вертикали и по горизонтали.
    return cv2.flip(image, 1)


def flip_vertical(image):
    # отразим изображение: 0 – по вертикали, 1 – по горизонтали, (-1) – по вертикали и по горизонтали.
    return cv2.flip(image, 0)


def flip_horizontal_and_vertical(image):
    # отразим изображение: 0 – по вертикали, 1 – по горизонтали, (-1) – по вертикали и по горизонтали.
    return cv2.flip(image, -1)


def get_flipped_images(image):
    return [image, flip_horizontal(image), flip_vertical(image), flip_horizontal_and_vertical(image)]


def start_loop(image, handlers, callback):
    if len(handlers) == 0:
        callback(image)
        return

    [current_handler, *rest_handlers] = handlers

    converted_images = current_handler(image)

    for current_image in converted_images:
        start_loop(current_image, rest_handlers, callback)


def show_callback(image):
    cv2.imshow(str(random.random()), image)


def get_save_to_file_callback(destination_images_path, global_index, extension):
    index = [0]

    def save_to_file_callback(image):
        saved_filename = f'{destination_images_path}/{global_index}_{index[0]}.{extension}'
        save_to_file(image, saved_filename)

        index[0] = index[0] + 1

    return save_to_file_callback


def get_source_image(source_image):
    return [source_image]


def handle_image(full_path, extension, global_index, destination_images_path, dimension, handlers):
    try:
        source_image = read_image(full_path, cv2.COLOR_RGB2GRAY)  # gray теперь падает на gray картинках
    except:
        source_image = read_image(full_path, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(source_image, dimension, interpolation=cv2.INTER_AREA)

    start_loop(resized, handlers, get_save_to_file_callback(destination_images_path, global_index, extension))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def handle_images_for_study(dimension, source_images_path, destination_images_path):
    # all handlers:
    # - get_images_with_filters
    # - get_cropped_and_rotated_images
    # - get_rotated_images
    # - get_flipped_images
    # - get_images_with_circles
    # - get_source_image

    handlers = [get_source_image]

    os.makedirs(destination_images_path)

    files_names = [f for f in listdir(source_images_path) if isfile(join(source_images_path, f))]

    index = 0

    for file_name in files_names:
        extension = file_name.split('.')[-1]
        full_path = f'{source_images_path}/{file_name}'

        handle_image(full_path, extension, index, destination_images_path, dimension, handlers)

        index = index + 1
        # print(f'Ready photo: {index} of {len(files_names)}')

    print(f'end: {source_images_path} -> {destination_images_path}')


def get_incorrect_dimensions(dimension, images_path):
    original_width, original_height = dimension
    files_names = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    print('Incorrect images:')

    for file_name in files_names:
        full_path = f'{images_path}/{file_name}'
        image = read_image(full_path, cv2.COLOR_RGB2BGR)  # COLOR_RGB2GRAY - не это, потому что падает
        width, height, _ = image.shape

        if (width != original_width) or (height != original_height):
            print(full_path)

    print('End of incorrect images')


def handle_images_for_predict(dimension, source_images_path, destination_images_path):
    # all handlers:
    # - get_images_with_filters
    # - get_cropped_and_rotated_images
    # - get_rotated_images
    # - get_flipped_images
    # - get_images_with_circles
    # - get_source_image

    handlers = [get_cropped_and_rotated_images]

    os.makedirs(destination_images_path)

    files_names = [f for f in listdir(source_images_path) if isfile(join(source_images_path, f))]
    index = 0

    for file_name in files_names:
        extension = file_name.split('.')[-1]
        full_path = f'{source_images_path}/{file_name}'

        handle_image(full_path, extension, index, destination_images_path, dimension, handlers)

        # source_image = read_image(full_path, cv2.COLOR_RGB2GRAY)
        # resized = cv2.resize(source_image, dimension, interpolation=cv2.INTER_AREA)
        #
        # saved_filename = f'{destination_images_path}/{index}.{extension}'
        # save_to_file(resized, saved_filename)

        index = index + 1

    print(f'End: {source_images_path} -> {destination_images_path}')
