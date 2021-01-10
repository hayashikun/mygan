import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from mygan.mnist import BATCH_SIZE, NOISE_DIM, \
    OUTPUT_DIR_PATH, LOG_FILE_PATH, CHECKPOINT_PREFIX, \
    CHECKPOINT_INTERVAL
from mygan.mnist.dataset import load_dataset
from mygan.mnist.model import make_generator, make_discriminator


class Trainer:
    def __init__(self):
        self.generator = make_generator()
        self.discriminator = make_discriminator()
        self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.fixed_noise = tf.random.normal([16, NOISE_DIM])

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.summary_writer = tf.summary.create_file_writer(LOG_FILE_PATH)
        self.generator_loss_metrics = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
        self.discriminator_loss_metrics = tf.keras.metrics.Mean("discriminator_loss", dtype=tf.float32)

        self.dataset = load_dataset()

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def generator_loss(self, fake_output):
        return self.criterion(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.criterion(tf.ones_like(real_output), real_output)
        fake_loss = self.criterion(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        self.generator_loss_metrics(gen_loss)
        self.discriminator_loss_metrics(disc_loss)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def save_generated_image(self, epoch):
        predictions = self.generator(self.fixed_noise, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            ax.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR_PATH, f"epoch_{epoch}"), dpi=300)
        plt.close()

    def train(self, epochs):
        for epoch in range(epochs):
            for image_batch in self.dataset:
                self.train_step(image_batch)

            with self.summary_writer.as_default():
                tf.summary.scalar('generator_loss', self.generator_loss_metrics.result(), step=epoch)
                tf.summary.scalar('discriminator_loss', self.discriminator_loss_metrics.result(), step=epoch)

            logging.info(f"Epoch {epoch + 1},"
                         f"\t Generator loss: {self.generator_loss_metrics.result()}"
                         f"\t Discriminator loss: {self.discriminator_loss_metrics.result()}")

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                self.checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

            self.generator_loss_metrics.reset_states()
            self.discriminator_loss_metrics.reset_states()
            self.save_generated_image(epoch + 1)
