from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import pickle
from datetime import datetime

from ImageProcessor import ImagePreprocessor
from ModelVisualizer import ModelVisualizer
from constants import BASE_DATA_SET_DIRECTORY, MODEL_HISTORY_DIRECTORY


def main():
    # Comment the following line if you want to display the newly trained model history instead
    history = load_history('ryan')

    # Uncomment the following line to train the model
    # history = train_model()

    # Display the accuracy of the model
    model_visualizer = ModelVisualizer()
    model_visualizer.plot_history_pickle_file(history)


def train_model():
    image_pre_processor = ImagePreprocessor(BASE_DATA_SET_DIRECTORY)

    train_generator = image_pre_processor.get_train_generator()
    validation_generator = image_pre_processor.get_validation_generator()

    # Defines a simplified CNN model for quick testing
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # Reduced filters
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),  # Reduced units
        Dense(1, activation='sigmoid')  # binary classification (normal/pneumonia)
    ])

    # Compiles the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    steps_per_epoch = len(train_generator.filenames) // train_generator.batch_size
    validation_steps = len(validation_generator.filenames) // validation_generator.batch_size

    # Trains the model with reduced epochs and steps for quick testing
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,  # Reduced for quick testing
        epochs=2,  # Reduced number of epochs for quick testing
        validation_data=validation_steps,
        validation_steps=5,  # Reduced for quick testing
        verbose=2  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    )

    # Save the history as a pickle file
    saved_path = save_history(history)

    return saved_path


def save_history(history):
    """
    Saves the history of the model to a pickle file
    """
    os.makedirs(MODEL_HISTORY_DIRECTORY, exist_ok=True)

    # Formats the current time as a string
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    filename = f'history_{timestamp}.pickle'
    full_path = os.path.join(MODEL_HISTORY_DIRECTORY, filename)

    with open(full_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f'History saved to {full_path}')

    return full_path


def load_history(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        history = pickle.load(f)
    return history


if __name__ == "__main__":
    main()
