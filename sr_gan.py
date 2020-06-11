import numpy as np
import tensorflow.keras.backend as K
import helper
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from data_loader import DataLoader
from generator import Generator
from discriminator import Discriminator
import cv2 


def vgg_loss(y, y_pred):
  vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  vgg19.trainable = False

  for layer in vgg19.layers:
    layer.trainable = False

  loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
  loss_model.trainable = False
  return K.mean(K.square(loss_model(y) - loss_model(y_pred)))

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

    # Create data loader object
    self.data_loader = DataLoader('data')

    dis, gen, gan = self.compile()
    self.train(dis, gen, gan)
  
  def compile(self):
     # Create network objects
    generator = Generator().get_model()
    discriminator = Discriminator(self.image_shape).get_model()

    # Create optimizer and compile networks
    generator.compile(loss=mean_squared_loss, optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

    # Print model summaries
    if self.verbose:
      generator.summary()
      discriminator.summary()
    
    gan = self._get_gan(discriminator, generator)
    return discriminator, generator, gan

  def _get_gan(self, discriminator, generator):
    """Returns the full combined network"""
    
    discriminator.trainable = False
    input_generator_gan = Input(shape=self.shape, name='input_generator_gan')
    output_generator_gan = generator(input_generator_gan)

    output_discriminator_gan = discriminator(output_generator_gan)

    generator_gan = Model(inputs=input_generator_gan, outputs=[output_generator_gan, output_discriminator_gan])

    generator_gan.compile(loss=mean_squared_loss,
                          loss_weights=[1., 1e-3],
                          optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

    if self.verbose:
      generator_gan.summary()
    
    return generator_gan

  def train(self, discriminator, generator, gan, epochs=10, batch_size=50):
    """Train the gan"""

    # Load the sets using the data loader
    hr_set = self.data_loader.get_hr_set()
    lr_set = self.data_loader.get_lr_set()

    set_count = len(hr_set)

    # Do a check if both sets are of equal length
    assert set_count == len(lr_set), "Image set must be of equal length"

    batch_count = int(set_count / batch_size)

    # Train for x number of epochs
    for epoch in range(1, epochs):
      helper.bprint(f"EPOCH: {epoch}")

      # Train on x random batches every epoch
      for _ in range(batch_count):
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

  def test(self):
    pass

  def save_weights(self, network):
    pass
