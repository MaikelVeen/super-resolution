import numpy as np
import cv2 
import os 

import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from data_loader import DataLoader
from generator import Generator
from discriminator import Discriminator
import helper

def vgg_54():
  return _vgg(20)

def _vgg(output_layer):
  vgg = VGG19(input_shape=(None, None, 3), include_top=False)
  vgg.trainable = False
  for l in vgg.layers:
    l.trainable = False
  return Model(vgg.input, vgg.layers[output_layer].output)

def mean_squared_loss(y_true, y_pred):
  return K.mean(K.square(y_pred - y_true), axis=-1)

class SRGAN():
  """ Class encapsulating the SR GAN network"""

  def __init__(self, verbose=True):
    np.random.seed(420)
    self.downscale_factor = 4   
    self.image_shape = (224, 224, 3)
    self.shape = (56, 56, 3)
    self.verbose = verbose

    self.mean_squared_error = MeanSquaredError()
    self.vgg = vgg_54()

    # Create data loader object
    self.data_loader = DataLoader('data')

    dis, gen, gan = self.compile()
    self.train(dis, gen, gan)

  def content_loss(self, y_pred, y_true):
    sr_features = self.vgg(y_pred)
    hr_features = self.vgg(y_true)

    return self.mean_squared_error(hr_features, sr_features)
  
  def compile(self):
     # Create network objects
    generator = Generator().get_model()
    discriminator = Discriminator(self.image_shape).get_model()

    # Create optimizer and compile networks
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=self.content_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    # Print model summaries
    if self.verbose:
      generator.summary()
      discriminator.summary()
    
    gan = self._get_gan(discriminator, generator, adam)
    return discriminator, generator, gan

  def _get_gan(self, discriminator, generator, optimizer):
    """Returns the full combined network"""
    
    discriminator.trainable = False
    input_generator_gan = Input(shape=self.shape, name='input_generator_gan')
    output_generator_gan = generator(input_generator_gan)

    output_discriminator_gan = discriminator(output_generator_gan)

    generator_gan = Model(inputs=input_generator_gan, outputs=[output_generator_gan, output_discriminator_gan])

    generator_gan.compile(loss=[self.content_loss, "binary_crossentropy"],
                          loss_weights=[1., 1e-3],
                          optimizer=optimizer)

    if self.verbose:
      generator_gan.summary()
  
    return generator_gan

  def train(self, discriminator, generator, gan, epochs=100, batch_size=20):
    """Train the gan"""

    # Load the sets using the data loader
    hr_set = self.data_loader.get_hr_set()
    lr_set = self.data_loader.get_lr_set()

    set_count = len(hr_set)

    # Do a check if both sets are of equal length
    assert set_count == len(lr_set), "Image set must be of equal length"

    batch_count = int(set_count / batch_size)

    # loss history lists
    real_losses = []
    fake_losses = []
    gan_losses = []

    # Train for x number of epochs
    for epoch in range(1, epochs):
      helper.bprint(f"EPOCH: {epoch}")

      # Train on x random batches every epoch
      for b in range(batch_count):
        rand = np.random.randint(0, set_count, size=batch_size)

        hr_batch = hr_set[rand]
        lr_batch = lr_set[rand]

        sr = generator.predict(lr_batch)

        real_Y = np.random.uniform(0.7, 1.2, size=batch_size).astype(np.float32)
        fake_Y = np.random.uniform(0.0, 0.3, size=batch_size).astype(np.float32)

        discriminator.trainable = True
        loss_real = discriminator.train_on_batch(hr_batch, real_Y)
        loss_fake = discriminator.train_on_batch(sr, fake_Y)
        discriminator.trainable = False

        gan_Y = np.ones((batch_size, 1), dtype = np.float32)
        loss_gan = gan.train_on_batch(lr_batch, [hr_batch, gan_Y])

        helper.gprint("Loss HR , Loss LR, Loss GAN")
        helper.gprint(f"{loss_real}, {loss_fake}, {loss_gan}")

        helper.print_progress(b, batch_count, epoch, loss_gan)
      
        if b == batch_count - 1:
          for i in range(0, 5):
            im = np.concatenate((self.data_loader.denormalize(hr_batch[i]), self.data_loader.denormalize(sr[i])), axis=1)
            cv2.imwrite(f"{os.path.dirname(os.path.abspath(__file__))}/result/e-{epoch}-{i}.png", im)


  def test(self):
    pass

  def save_weights(self, network):
    pass
