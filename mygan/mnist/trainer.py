import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from mygan.mnist import BATCH_SIZE, NOISE_DIM, \
    OUTPUT_DIR_PATH, LOG_FILE_PATH, CHECKPOINT_PREFIX, \
    CHECKPOINT_INTERVAL
from mygan.mnist.dataset import load_dataset
from mygan.mnist.model import make_generator, make_discriminator

generator = make_generator()
discriminator = make_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_loss_metrics = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
discriminator_loss_metrics = tf.keras.metrics.Mean("discriminator_loss", dtype=tf.float32)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_loss_metrics(gen_loss)
    discriminator_loss_metrics(disc_loss)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))


fixed_noise = tf.random.normal([16, NOISE_DIM])


def save_generated_image(epoch):
    predictions = generator(fixed_noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR_PATH, f"epoch_{epoch}"), dpi=300)
    plt.close()


dataset = load_dataset()

summary_writer = tf.summary.create_file_writer(LOG_FILE_PATH)


def train(epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        with summary_writer.as_default():
            tf.summary.scalar('generator_loss', generator_loss_metrics.result(), step=epoch)
            tf.summary.scalar('discriminator_loss', discriminator_loss_metrics.result(), step=epoch)

        logging.info(f"Epoch {epoch + 1},"
                     f"\t Generator loss: {generator_loss_metrics.result()}"
                     f"\t Discriminator loss: {discriminator_loss_metrics.result()}")

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        generator_loss_metrics.reset_states()
        discriminator_loss_metrics.reset_states()
        save_generated_image(epoch + 1)
