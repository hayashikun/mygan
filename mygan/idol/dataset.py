import glob
import os

import tensorflow as tf


def _load_image(path, image_size):
    raw = tf.io.read_file(path)
    tensor = tf.image.decode_image(raw)
    data = tf.image.resize(tensor, [image_size, image_size])
    data /= 255.0
    return data


def load_dataset(data_path, batch_size, image_size):
    files = glob.glob(os.path.join(data_path, "*.jpg"))
    data = [_load_image(f, image_size) for f in files]
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(files)).batch(batch_size)
    return dataset
