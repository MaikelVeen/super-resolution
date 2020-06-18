import os
import time
import curses
import sys
import cv2
import numpy as np

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import tensorflow.keras.losses
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects

from image_batch_loader import ImageBatchLoader
from generator import Generator
from discriminator import Discriminator
import helper

SAVE_SETTINGS =  {
            'title': 'Super Resolution',
            'tags': ['LR', 'SR', 'HR'],
            'text': {
              'font_color': (255, 255, 255),
              'border_color': (0, 0, 0),
              'font_size': 0.7,
              'font_thickness': 2,
              'border_thickness': 3,
            }
          }


class SRGAN():
  """ Class encapsulating the SR GAN network"""

  def __init__(self, verbose=True):
    # TODO: get these from argument list or config file
    self.downscale_factor = 4
    self.image_shape = (224, 224, 3)
    self.shape = (56, 56, 3)
    self.verbose = verbose
    self.batch_size = 20
    self.max_bath_size = 1500
    self.mean_squared_error = MeanSquaredError()
    self.vgg = self._get_vgg()

    # Create data loader object
    self.batch_loader = ImageBatchLoader(self.batch_size)

    #dis, gen, gan = self.compile()
    #self.train(dis, gen, gan)

  def _content_loss(self, y_pred, y_true):
    sr_features = self.vgg(y_pred)
    hr_features = self.vgg(y_true)

    return self.mean_squared_error(hr_features, sr_features)
  
  def compile(self):
     # Create network objects
    generator = Generator().get_model()
    discriminator = Discriminator(self.image_shape).get_model()

    # Create optimizer and compile networks
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=self._content_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    # Print model summaries
    if self.verbose:
      generator.summary()
      discriminator.summary()
    
    gan = self._get_gan(discriminator, generator, adam)
    return discriminator, generator, gan

  def load_models(self, gan_path, gen_path, dis_path):
    self.generator = load_model(gen_path, custom_objects={'_content_loss': self._content_loss})
    self.generator.summary()
    
  def _get_gan(self, discriminator, generator, optimizer):
    """Returns the full combined network"""
    
    discriminator.trainable = False
    input_generator_gan = Input(shape=self.shape, name='input_generator_gan')
    output_generator_gan = generator(input_generator_gan)

    output_discriminator_gan = discriminator(output_generator_gan)

    generator_gan = Model(inputs=input_generator_gan, outputs=[output_generator_gan, output_discriminator_gan])

    generator_gan.compile(loss=[self._content_loss, "binary_crossentropy"],
                          loss_weights=[1., 1e-3],
                          optimizer=optimizer)

    if self.verbose:
      generator_gan.summary()
  
    return generator_gan

  def train(self, discriminator, generator, gan, batch_size=20, epochs=100):
    """Train the gan"""
    batch_count = int(36900 / batch_size)
    
    stdscr = curses.initscr()
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.halfdelay(1)
    curses.noecho()

    # loss history lists
    real_losses = []
    fake_losses = []
    gan_losses = []

    if self.max_bath_size < batch_count:
      batch_count = self.max_bath_size

    # Train for x number of epochs
    for epoch in range(1, epochs):
      self.batch_loader.reset()
      # Train on x random batches every epoch
      for batch in range(batch_count):
        hr_batch, lr_batch = self.batch_loader.next_batch()
        sr_batch = generator.predict(lr_batch)

        real_y = np.random.uniform(0.7, 1.2, size=batch_size).astype(np.float32)
        fake_y = np.random.uniform(0.0, 0.3, size=batch_size).astype(np.float32)

        discriminator.trainable = True
        loss_real = discriminator.train_on_batch(hr_batch, real_y)
        loss_fake = discriminator.train_on_batch(sr_batch, fake_y)
        discriminator.trainable = False

        gan_y = np.ones((batch_size, 1), dtype=np.float32)
        loss_gan = gan.train_on_batch(lr_batch, [hr_batch, gan_y])

        real_losses.append(loss_real)
        fake_losses.append(loss_fake)
        gan_losses.append(loss_gan)

        helper.print_progress_bar(stdscr, batch, batch_count, epoch, epochs, loss_real, loss_fake, loss_gan, True)

        if batch == 10 - 1:
          
          # TODO maybe also move this.
          for i in range(0, 5):
            y_hat = self.batch_loader.denormalize(sr_batch[i])
            y = self.batch_loader.denormalize(hr_batch[i])
            x = self.batch_loader.denormalize(lr_batch[i])
            x = cv2.resize(x, (0, 0), fx=4, fy=4)
            helper.save_result(f"{os.path.dirname(os.path.abspath(__file__))}/result/e-{epoch}-{i}.png", [x, y_hat, y], SAVE_SETTINGS)

      if epoch % 5 == 0:
        gan.save(f"gan-e{epoch}.h5")
        discriminator.save(f"dis-e{epoch}.h5")
        generator.save(f"gen-e{epoch}.h5")

    helper.plot_loss(real_losses, fake_losses, gan_losses)
    curses.endwin()

  def predict(self, file_paths_lr, file_paths_hr, filenames):
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
        [lr_image, sr_image, hr_image], SAVE_SETTINGS)

  def _save_weights(self, network):
    pass
  
  @staticmethod
  def _get_vgg():
    # Returns a non-trainable vgg Model
    vgg_net = VGG19(input_shape=(None, None, 3), include_top=False)
    vgg_net.trainable = False
    for layer in vgg_net.layers:	
        layer.trainable = False
    return Model(vgg_net.input, vgg_net.layers[20].output)

