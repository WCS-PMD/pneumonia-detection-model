import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Define model
img_size = 128

# Declare preprocessing functions
tf.random.set_seed(999)
img_size = 128


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)

    # Check the number of channels
    num_channels = tf.shape(img)[-1]

    # Convert grayscale to RGB if necessary
    if num_channels == 1:
        print("grayscale")
        img = np.repeat(img, 3, axis=-1)
    # get image dimensions
    img_shape = tf.shape(img).numpy()

    width, height = img_shape[0], img_shape[1]
    min_dimension = tf.math.minimum(width, height).numpy()

    crop_size = tf.constant([min_dimension, min_dimension, 3], dtype='int32')
    # Implement random cropping
    img = tf.image.random_crop(value=img, size=crop_size)
    img = tf.cast(img, tf.float32) / 255  # Explicitly cast to float32 and normalize
    return tf.image.resize(img, [img_size, img_size], method='nearest', preserve_aspect_ratio=False)


@tf.function
def process_image(file_storage):
    img_content = file_storage.read()
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

model.load_weights('xray_model_90_percent.keras')

app = Flask(__name__)
CORS(app)  # Enable CORS


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image = process_image(file)
            image = tf.expand_dims(image, 0)

            # Make a prediction
            predictions = model.predict(image)
            predicted_class = class_names[np.argmax(predictions)]

            return jsonify({'prediction': predicted_class})
        except Exception as e:
            # For debugging purposes
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
