import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from mygan import TmpFilePath
from mygan.idol.dataset import load_dataset
from mygan.idol.model import make_generator, make_discriminator

SAVE_IMAGE_INTERVAL = 10
CHECKPOINT_INTERVAL = 100


class Trainer:
    def __init__(self, tag, data_path, output_path=None):
        self.epoch = 0
        self.batch_size = 100
        self.noise_dim = 256
        self.generator = make_generator(self.noise_dim)
        self.discriminator = make_discriminator()
        self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.fixed_noise = tf.random.normal([16, self.noise_dim])

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        if output_path is not None:
            self.output_path = os.path.join(output_path, tag)
        else:
            self.output_path = os.path.join(TmpFilePath, "idol", tag)

        self.log_file_dir = os.path.join(self.output_path, "log")
        self.checkpoint_dir = os.path.join(self.output_path, "checkpoint")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint")

        for d in [self.output_path, self.checkpoint_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        self.checkpoint_epoch_log_path = os.path.join(self.log_file_dir, "epoch")

        if os.path.exists(self.checkpoint_epoch_log_path):
            with open(self.checkpoint_epoch_log_path) as fp:
                self.epoch = int(fp.read())
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        self.summary_writer = tf.summary.create_file_writer(self.log_file_dir)
        self.generator_loss_metrics = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
        self.discriminator_loss_metrics = tf.keras.metrics.Mean("discriminator_loss", dtype=tf.float32)

        self.dataset = load_dataset(data_path, self.batch_size, 64)

    def generator_loss(self, fake_output):
        return self.criterion(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.criterion(tf.ones_like(real_output), real_output)
        fake_loss = self.criterion(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        self.generator_loss_metrics(gen_loss)
        self.discriminator_loss_metrics(disc_loss)

        self.generator_optimizer.apply_gradients(
            zip(
                gen_tape.gradient(gen_loss, self.generator.trainable_variables),
                self.generator.trainable_variables
            )
        )
        self.discriminator_optimizer.apply_gradients(
            zip(
                disc_tape.gradient(disc_loss, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables
            )
        )

    def save_generated_image(self, epoch):
        predictions = self.generator(self.fixed_noise, training=False)
        predictions = predictions.numpy()
        predictions[predictions < 0] = 0
        predictions[predictions > 1] = 1

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.imshow(predictions[i])
            ax.axis("off")

        fig.tight_layout(pad=0.2)
        fig.savefig(os.path.join(self.output_path, f"epoch_{epoch}.jpg"), dpi=150)
        plt.close()

    def save_checkpoint(self, epoch):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        with open(self.checkpoint_epoch_log_path, 'w') as fp:
            fp.write(str(epoch))

    def train(self, epochs):
        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch + epochs):
            for images in self.dataset:
                self.train_step(images)

            with self.summary_writer.as_default():
                tf.summary.scalar("generator_loss", self.generator_loss_metrics.result(), step=epoch)
                tf.summary.scalar("discriminator_loss", self.discriminator_loss_metrics.result(), step=epoch)

            logging.info(f"Epoch {epoch + 1},"
                         f"\t Generator loss: {self.generator_loss_metrics.result()}"
                         f"\t Discriminator loss: {self.discriminator_loss_metrics.result()}")

            if (epoch + 1) % SAVE_IMAGE_INTERVAL == 0:
                self.save_generated_image(epoch + 1)

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(epoch + 1)

            self.generator_loss_metrics.reset_states()
            self.discriminator_loss_metrics.reset_states()

            self.epoch = epoch

