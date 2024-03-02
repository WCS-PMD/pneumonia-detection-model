from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
tf.get_logger().setLevel('INFO')

app = Flask(__name__)
CORS(app)  # Enable CORS


@app.route('/')
def hello_world():
    return 'Hello from Flask!'


# Define model
img_size = 128

# Declare preprocessing functions
tf.random.set_seed(999)
img_size = 128


def decode_img(img):
    logging.info("Starting to decode and process image")

    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)

    logging.info(f"Decoded image shape: {img.shape}")

    # Check the number of channels
    num_channels = tf.shape(img)[-1]

    # Convert grayscale to RGB if necessary
    if num_channels == 1:
        logging.info("Image is grayscale, converting to RGB")
        img = np.repeat(img, 3, axis=-1)
    # get image dimensions
    img_shape = tf.shape(img).numpy()

    width, height = img_shape[0], img_shape[1]
    min_dimension = tf.math.minimum(width, height).numpy()

    crop_size = tf.constant([min_dimension, min_dimension, 3], dtype='int32')
    # Implement random cropping
    img = tf.image.random_crop(value=img, size=crop_size)
    img = tf.cast(img, tf.float32) / 255  # Explicitly cast to float32 and normalize

    logging.info("Image processing completed")

    return tf.image.resize(img, [img_size, img_size], method='nearest', preserve_aspect_ratio=False)


@tf.function
def process_image(file_storage):
    logging.info("Reading image file")
    img_content = file_storage.read()
    logging.info("File read successfully, processing image now")
    img = tf.py_function(func=decode_img, inp=[img_content], Tout=tf.float32)
    return img


class_names = ['COVID-19', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral']

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(len(class_names), activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.load_weights('/home/feernen/mysite/xray_model_90_percent.keras')


@app.route('/predict', methods=['GET'])
@cross_origin()
def say_hi():
    return 'Hi! You\'ve made a get request to the predict endpoint.'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    logging.info("Received a prediction request")
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            logging.info("Processing image for prediction")
            image = process_image(file)
            image = tf.expand_dims(image, 0)

            # Make a prediction
            predictions = model.predict(image)
            predicted_class = class_names[np.argmax(predictions)]

            logging.info(f"Prediction made: {predicted_class}")
            return jsonify({'prediction': predicted_class})
        except Exception as e:
            logging.exception("Error making prediction")
            # For debugging purposes
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()
