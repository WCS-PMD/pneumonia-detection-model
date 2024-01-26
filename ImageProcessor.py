import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessor:
    """
    Sets up an ImageDataGenerator which will perform the preprocessing operations in-memory on-the-fly as images
    are loaded into the model.

    Usage:
        base_dir = 'chest_xray'
        preprocessor = ImagePreprocessor(base_dir)
        train_generator = preprocessor.get_train_generator()
        validation_generator = preprocessor.get_validation_generator()
    """

    def __init__(self, base_dir, image_size=(150, 150), batch_size=32):
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_dir = f'{self.base_dir}/train'
        self.validation_dir = f'{self.base_dir}/val'
        self.test_dir = f'{self.base_dir}/test'

        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.val_datagen = ImageDataGenerator(rescale=1. / 255)

    def get_train_generator(self):
        return self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

    def get_validation_generator(self):
        return self.val_datagen.flow_from_directory(
            self.validation_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
