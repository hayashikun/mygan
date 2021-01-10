from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_path, batch_size, image_size):
    data_generator = ImageDataGenerator(
        rescale=1. / 255,
    )
    return data_generator.flow_from_directory(batch_size=batch_size,
                                              directory=data_path,
                                              shuffle=True,
                                              target_size=(image_size, image_size),
                                              class_mode='binary')
