import cv2
import os
import tensorflow as tf
from tensorflow.keras import models, layers
import pathlib
import numpy as np
import matplotlib.pyplot as plt

class pneumonia_detection_model(models.Sequential()):
    @tf.function
    def train_step(data):
        print('train step')
        images, labels = data

        print("Image shape:", images.shape)
        print("Label shape:", labels.shape)

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

        print("Loss:", loss.numpy())  # Add this line to print the loss during training


def get_label(file_path: str):
    parsed = tf.strings.split(file_path, os.path.sep)
    parent = parsed[-2]
    file_name = parsed[-1]
    if parent == "PNEUMONIA":
        label_str = tf.strings.split(file_name, '_')[1]
    else:
        label_str = parent
    return tf.argmax(label_str == class_names)


def decode_img(img):
    img_size = 64
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img, channels=3, dtype= tf.dtypes.uint8)

    # Check the number of channels
    num_channels = tf.shape(img)[-1]

    # Convert RGB to grayscale if necessary
    if num_channels == 3:
        img = tf.image.rgb_to_grayscale(img)
    # get image dimensions
    img_shape = tf.shape(img).numpy()

    width, height = img_shape[0], img_shape[1]
    min_dimension = tf.math.minimum(width, height).numpy()

    crop_size = tf.constant([min_dimension, min_dimension, 1], dtype='int32')
    # Implement random cropping
    img = tf.image.random_crop(value=img, size=crop_size)
    img = tf.cast(img, tf.uint8)  # Explicitly cast to uint8
    return tf.image.resize(img, [img_size, img_size], method='nearest', preserve_aspect_ratio=False)


@tf.function
def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.py_function(decode_img, [img], tf.uint8)
    return img, label




if __name__ == "__main__":
    path_to_dir = os.environ.get("DATASET_PATH")

    if path_to_dir is None:
        os.environ["DATASET_PATH"] = input("Enter complete path to the chest_xray folder: ")
        path_to_dir = os.environ["DATASET_PATH"]
        print("You should set the DATASET_PATH enviroment variable to avoid this every time you run the program.")

    ds_files = [str(file.absolute()) for file in pathlib.Path(path_to_dir).glob("**/[!._]*.jpeg")]

    list_ds = tf.data.Dataset.from_tensor_slices(ds_files)
    image_count = len(ds_files)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    for f in list_ds.take(5):
        print(f.numpy())

    class_names = np.array(['NORMAL', 'bacteria', 'virus'])

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    for image, label in train_ds.take(3):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())
        print("Label Shape: ", label.numpy().shape)
        print("Type: ", class_names[label.numpy()])
        plt.imshow(image.numpy())
        plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=10,
                        validation_data=val_ds)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(val_ds, verbose=2)










