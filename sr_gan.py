import os
import curses
import cv2
import numpy as np
import yaml
import pickle

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from image_batch_loader import ImageBatchLoader
from generator import Generator
from discriminator import Discriminator
import helper


class SRGAN():
    """ Class encapsulating the SR GAN network"""

    def __init__(self):
        self.config = self._load_config()

        # Set network objects to None
        # Can be created using compile or load methods
        self.gan = None
        self.generator = None
        self.discriminator = None

        self.mean_squared_error = MeanSquaredError()
        self.vgg = self._get_vgg()
        self.batch_loader = ImageBatchLoader(self.config['batch']['size'])
        self.set_size = self.batch_loader.set_size

    def compile(self):
        """Builds and compiles the discriminator, generator and gan"""
        self.generator = Generator().get_model()
        shape = (self.config['hr_shape'], self.config['hr_shape'], 3)
        self.discriminator = Discriminator(shape).get_model()

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.generator.compile(loss=self._content_loss, optimizer=adam)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=adam)

        # Print model summaries
        if self.config['verbose']:
            self.generator.summary()
            self.discriminator.summary()

        self._compile_gan(adam)

    def load_weights(self, epoch, subfolder=''):
        """"Loads the saved weights from the filesystem"""
        path = f"{os.path.dirname(os.path.abspath(__file__))}/weights/{subfolder}"

        generator_path = f"{path}{self.config['model_saving']['weight_saving']['gen_filename']}e{epoch}.h5"
        self.generator.load_weights(generator_path)

        discriminator_path = f"{path}{self.config['model_saving']['weight_saving']['dis_filename']}e{epoch}.h5"
        self.discriminator.load_weights(discriminator_path)
        
        gan_path = f"{path}{self.config['model_saving']['weight_saving']['gan_filename']}e{epoch}.h5"
        self.gan.load_weights(gan_path)

    def load_optimizers(self, epoch):
        """"Loads the saved optimizers from the filesystem"""
        path = f"{os.path.dirname(os.path.abspath(__file__))}/optimizers/"

        with open(f"{path}{self.config['model_saving']['optimizer_saving']['gen_filename']}e{epoch}.pkl", 'rb') as f:
            weight_values = pickle.load(f)
        self.generator.optimizer.set_weights(weight_values)

        with open(f"{path}{self.config['model_saving']['optimizer_saving']['dis_filename']}e{epoch}.pkl", 'rb') as f:
            weight_values = pickle.load(f)
        self.discriminator.optimizer.set_weights(weight_values)

        with open(f"{path}{self.config['model_saving']['optimizer_saving']['gan_filename']}e{epoch}.pkl", 'rb') as f:
            weight_values = pickle.load(f)
        self.gan.optimizer.set_weights(weight_values)

    def train(self, epoch_start=0, load_only=False):
        """Training loop for the generative adversarial network"""
        self._pre_train_check()

        # Get batch size and calculate the batches per epoch
        batch_size = self.config['batch']['size']
        batch_count = min(int(self.set_size / batch_size), self.config['batch']['max_per_epoch'])
        epochs = self.config['epochs'] + epoch_start

        real_losses = []
        fake_losses = []
        gan_losses = []

        stdscr = self._init_printer()

        # Train for x number of epochs
        for epoch in range(1 + epoch_start, epochs + 1):
            self.batch_loader.reset()

            # Train on x random batches every epoch
            for batch in range(batch_count):
                # Get all batches from batch loader and generator
                hr_batch, lr_batch = self.batch_loader.next_batch()
                sr_batch = self.generator.predict(lr_batch)

                if self.config['model_loading']['active'] is True and epoch == 1 + epoch_start and batch_count == 1:
                    self.load_optimizers(self.config['model_loading']['epoch'])

                if load_only:
                    break

                # Create sliding prediction array
                real_y = np.random.uniform(0.7, 1.2, size=batch_size).astype(np.float32)
                fake_y = np.random.uniform(0.0, 0.3, size=batch_size).astype(np.float32)

                # Train discriminator
                self.discriminator.trainable = True
                loss_real = self.discriminator.train_on_batch(hr_batch, real_y)
                loss_fake = self.discriminator.train_on_batch(sr_batch, fake_y)
                self.discriminator.trainable = False

                # Train gan
                gan_y = np.ones((batch_size, 1), dtype=np.float32)
                loss_gan = self.gan.train_on_batch(lr_batch, [hr_batch, gan_y])

                real_losses.append(loss_real)
                fake_losses.append(loss_fake)
                gan_losses.append(loss_gan)

                helper.print_progress_bar(stdscr, batch, batch_count, epoch, epochs, loss_real, loss_fake, loss_gan, True)

                if batch == batch_count - 1:
                    self._save_samples(lr_batch, sr_batch, hr_batch, epoch)

            if self.config['model_saving']['active'] is True and epoch % self.config['model_saving']['after_epoch'] == 0:
                self._save_weights(epoch)
                self._save_optimizers(epoch)
            
            if load_only:
                    break

        curses.endwin()
        #helper.plot_loss(real_losses, fake_losses, gan_losses)

    def predict(self, file_paths_lr, file_paths_hr, filenames):
        """Generates predictions for given images"""
        helper.bprint("Loading image batches")
        hr_images = self.batch_loader.load_images(file_paths_hr)
        lr_images = self.batch_loader.load_images(file_paths_lr)

        helper.bprint("Upscaling low resolution images")
        sr_images = self.generator.predict(lr_images)

        helper.gprint("Saving results")
        image_count = len(lr_images)

        for i in range(0, image_count):
            helper.cprint(f"Denormalizing image {i}")
            lr_image = cv2.resize(self.batch_loader.denormalize(lr_images[i]), (0, 0), fx=4, fy=4)
            hr_image = self.batch_loader.denormalize(hr_images[i])
            sr_image = self.batch_loader.denormalize(sr_images[i])

            helper.save_result(f"{os.path.dirname(os.path.abspath(__file__))}/result/{filenames[i]}.png", 
                [lr_image, sr_image, hr_image])
    
    def evaluate(self):
        """Calculates and prints the mean pnsr over validation data set"""
        np.random.seed(420)
        helper.gprint('Evaulating validation set')
        self.train(load_only=True)
        eval_batch_loader = ImageBatchLoader(batch_size=1, hr='/validation_hr', lr='/validation_lr')

        nearest_metrics = []
        cubic_metrics = []
        sr_metrics = []

        for i in range(0, eval_batch_loader.set_size):
            print(f"Evaluating {i}")
            hr_batch, lr_batch = eval_batch_loader.next_batch()
            sr_batch = self.generator.predict(lr_batch)

            lr_image = eval_batch_loader.denormalize(lr_batch[0])
            hr_image = eval_batch_loader.denormalize(hr_batch[0])
            sr_image = eval_batch_loader.denormalize(sr_batch[0])

            nearest = cv2.resize(lr_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cubic = nearest = cv2.resize(lr_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

            nearest_metrics.append(tf.image.psnr(nearest, hr_image, max_val=255))
            cubic_metrics.append(tf.image.psnr(cubic, hr_image, max_val=255))
            sr_metrics.append(tf.image.psnr(sr_image, hr_image, max_val=255)) 
        
        print(f"Mean of nearest neighbour interpolation PSNR: {np.mean(nearest_metrics)}")
        print(f"Mean of cubic interpolation PSNR: {np.mean(cubic_metrics)}")
        print(f"Mean of SRGAN PSNR: {np.mean(sr_metrics)}")

    
    def _compile_gan(self, optimizer):
        # Combines the discriminator and generator and compiles the gan
    
        self.discriminator.trainable = False
        shape = (self.config['lr_shape'], self.config['lr_shape'], 3)
        input_generator_gan = Input(shape=shape, name='input_generator_gan')
        output_generator_gan = self.generator(input_generator_gan)
        output_discriminator_gan = self.discriminator(output_generator_gan)

        generator_gan = Model(inputs=input_generator_gan, outputs=[output_generator_gan, output_discriminator_gan])
        generator_gan.compile(loss=[self._content_loss, "binary_crossentropy"],
                            metrics=[self._psnr],
                            loss_weights=[1., 1e-3],
                            optimizer=optimizer)
        
        if self.config['verbose']:
            generator_gan.summary()

        self.gan = generator_gan

    def _pre_train_check(self):
        # Check if all the models are not None
        error_message = self.config['empty_model_error']
        assert self.generator is not None, error_message
        assert self.discriminator is not None, error_message
        assert self.gan is not None, error_message

    def _save_weights(self, epoch):
        # Save the weight of all the models to the filesystem
        path = f"{os.path.dirname(os.path.abspath(__file__))}/weights/"
        self.generator.save_weights(f"{path}{self.config['model_saving']['weight_saving']['gen_filename']}e{epoch}.h5")
        self.discriminator.save_weights(f"{path}{self.config['model_saving']['weight_saving']['dis_filename']}e{epoch}.h5")
        self.gan.save_weights(f"{path}{self.config['model_saving']['weight_saving']['gan_filename']}e{epoch}.h5")

    def _save_optimizers(self, epoch):
        # Save the weights of all the optimizers to the filesystem
        path = f"{os.path.dirname(os.path.abspath(__file__))}/optimizers/"

        worklist = [
            self.generator,
            self.discriminator,
            self.gan
        ]

        filenames = [
            self.config['model_saving']['optimizer_saving']['gen_filename'],
            self.config['model_saving']['optimizer_saving']['dis_filename'],
            self.config['model_saving']['optimizer_saving']['gan_filename']
        ]

        for network, filename in zip(worklist, filenames):
            symbolic_weights = getattr(network.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(f"{path}{filename}e{epoch}.pkl", 'wb') as f:
                pickle.dump(weight_values, f)

    def _save_samples(self, lr_batch, sr_batch, hr_batch, epoch):
        # Saves x samples from the current batch
        count = self.config['sample_saving']['count']
        filename = self.config['sample_saving']['filename']

        for i in range(0, count):
            sr_image = self.batch_loader.denormalize(sr_batch[i])
            hr_image = self.batch_loader.denormalize(hr_batch[i])
            lr_image = self.batch_loader.denormalize(lr_batch[i])
            lr_upscaled = cv2.resize(lr_image, (0, 0), fx=4, fy=4)

            helper.save_result(f"/{filename}{epoch}-{i}.png", [lr_upscaled, sr_image, hr_image])

    def _content_loss(self, y_pred, y_true):
        # Returns computed content loss
        sr_features = self.vgg(y_pred)
        hr_features = self.vgg(y_true)
        return self.mean_squared_error(hr_features, sr_features)
    
    def _psnr(self, y_true, y_pred):
        max_pixel = 255.0
        return 10.0 * K.log((max_pixel ** 2) / self.mean_squared_error(y_true, y_pred))

    @staticmethod
    def _get_vgg():
        # Returns a non-trainable vgg Model
        vgg_net = VGG19(input_shape=(None, None, 3), include_top=False)
        vgg_net.trainable = False
        for layer in vgg_net.layers:
            layer.trainable = False
        return Model(vgg_net.input, vgg_net.layers[20].output)
                
    @staticmethod
    def _load_config():
        # Loads the configuration variables from yaml
        config_path = f"{os.path.dirname(os.path.abspath(__file__))}/config.yaml"
        return yaml.safe_load(open(config_path))
    
    @staticmethod
    def _init_printer():
        stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.halfdelay(1)
        curses.noecho()
        return stdscr

    
