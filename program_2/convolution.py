import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf

keras = tf.keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs available: ", len(gpus))

def main():
    TRAINING_DIR = 'dataset'
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = 'dataset-test'
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        class_mode='categorical'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    class TrainCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return

            if logs.get('accuracy') > 0.98:
                print("\nReached 98% accuracy. Stopping training...")
                self.model.stop_training = True

    callback = TrainCallback()

    history = model.fit(
        train_generator,
        steps_per_epoch=200,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=10,
        verbose=1,
    )

    model.save("model")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()


main()
