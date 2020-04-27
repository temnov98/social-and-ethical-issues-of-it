import pathlib

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

keras = tf.keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs available: ", len(gpus))

CLASS_NAMES = ['barash', 'car-carich', 'copatich', 'crosh', 'ejik', 'losyash', 'nusha', 'pin', 'sovunja']

model: keras.models.Sequential = keras.models.load_model('../program_2/model')

model.summary()

print(CLASS_NAMES)

data_dir = pathlib.Path('predict-examples')
for path in data_dir.glob('*'):
    img = tf.io.read_file(str(path))
    image_decoded = tf.image.decode_png(img, channels=3)

    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0

    prediction = model.predict_classes(image)
    print('File: ', str(path))
    print(prediction)
    print(CLASS_NAMES[prediction[0]])
    print('_____________________________________')


# print('Result: ', CLASS_NAMES[prediction[0]])
