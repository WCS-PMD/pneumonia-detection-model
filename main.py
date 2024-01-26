import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from ImageProcessor import ImagePreprocessor
from ModelVisualizer import ModelVisualizer
from constants import BASE_DATA_SET_DIRECTORY


def main():
    image_pre_processor = ImagePreprocessor(BASE_DATA_SET_DIRECTORY)

    train_generator = image_pre_processor.get_train_generator()
    validation_generator = image_pre_processor.get_validation_generator()

    # Defines a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')  # binary classification (normal/pneumonia)
    ])

    # Compiles the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Trains the model
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        # Number of batches per epoch. Supposedly this should be set to the number of samples divided by the batch size.
        epochs=15,  # Number of epochs to train for.
        validation_data=validation_generator,
        validation_steps=50,  # Number of validation batches to evaluate at the end of each epoch.
        verbose=2  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    )

    # Display the accuracy of the model
    model_visualizer = ModelVisualizer()
    model_visualizer.plot_history(history)


if __name__ == "__main__":
    main()
