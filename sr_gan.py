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

def vgg_loss(image_shape, y, y_pred):
  vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
  vgg19.trainable = False
 
  for layer in vgg19.layers:
    layer.trainable = False

  loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
  loss_model.trainable = False
  return K.mean(K.square(loss_model(y) - loss_model(y_pred)))

class SRGAN():
  """ Class encapsulating the SR GAN network"""

  def __init__(self, verbose=True):
    np.random.seed(420)
    self.downscale_factor = 4   
    self.image_shape = (224, 224, 3)
    self.shape = (56, 56, 3)
    self.verbose = verbose

    # Create data loader object
    self.data_loader = DataLoader('data', size=224,
                                  downscale_factor=self.downscale_factor, 
                                  generate_data=False)

    # Create network objects
    generator = Generator().get_model()
    discriminator = Discriminator(self.image_shape).get_model()

    # Create optimizer and compile networks
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    # Print model summaries
    if self.verbose:
      generator.summary()
      discriminator.summary()
    
    self.network = self.get_gan(discriminator, generator, adam)

  def get_gan(self, discriminator, generator, optimizer):
    """Returns the full combined network"""

    input_generator_gan = Input(shape=self.shape, name='input_generator_gan')
    output_generator_gan = generator(input_generator_gan)

    output_discriminator_gan = discriminator(output_generator_gan)

    generator_gan = Model(inputs=input_generator_gan, 
    outputs=[output_generator_gan, output_discriminator_gan])

    generator_gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                          loss_weights=[1., 1e-3],
                          optimizer=optimizer)

    if self.verbose:
      generator_gan.summary()
    
    return generator_gan


  def train(self, epochs, batch_size):
    pass

  def test(self):
    pass

  
