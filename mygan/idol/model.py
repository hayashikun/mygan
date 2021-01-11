from tensorflow.keras import layers, Sequential


def make_generator(noise_dim):
    model = Sequential()

    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512)

    model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding="same", use_bias=False))
    assert model.output_shape == (None, 64, 64, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 128, 128, 3)

    return model


_ = make_generator(1)


def make_discriminator():
    model = Sequential()

    model.add(layers.Conv2D(32, kernel_size=5, strides=2, padding="same", input_shape=(128, 128, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 64, 64, 32)

    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 32, 32, 64)

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 16, 16, 128)

    model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 8, 8, 256)

    model.add(layers.Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 4, 4, 512)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


_ = make_discriminator()
