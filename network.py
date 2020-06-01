
import numpy as np
from generator import Generator
from discriminator import Discriminator
from data_loader import DataLoader
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class SRGAN():
  def __init__(self):
    np.random.seed(10)
    downscale_factor = 4
    image_shape = (224, 224, 3)
    shape = (56, 56, 3)

    self.data_loader = DataLoader('data', size=224, downscale_factor=downscale_factor)

    #gen = Generator((64,64,3))
    #self.generator = gen.get()

    #dis = Discriminator((64,64,3))
    #self.discriminator = dis.get()

    # self.generator.summary()
    # self.discriminator.summary()

  def get_gan(self, dis, gen, shape, opt):
    pass

  def train(self, epochs, batch_size):
    pass

  def test(self):
    pass

  def vgg_loss(self, image_shape, y, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False

    for layer in vgg19.layers:
      layer.trainable = False

    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False

    return K.mean(K.square(loss_model(y) - loss_model(y_pred)))
