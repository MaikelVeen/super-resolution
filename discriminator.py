from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import helper


class Discriminator():
  def __init__(self, input_shape, res_blocks=16, upsampling=2):
    self.input_shape = input_shape
    self.upsampling = upsampling
    self.res_blocks = res_blocks

    self.discriminator = self._build_model()

  def get_model(self):
    """Returns the model object"""
    return self.discriminator

  def _dis_block(self, model, filters=64, kernel_size=3, strides=1):
    """ Adds a residual block to the sequential model """

    model = Conv2D(filters=filters, kernel_size=kernel_size,
                     strides=strides, padding="same")(model)
    #model.add(BatchNormalization(momentum=0.5))
    model = LeakyReLU(alpha=0.2)(model)
    return model

  def _build_model(self):
    """Create the model"""
    helper.cprint("Building discrimininator network")

    model_input = Input(shape=self.input_shape)

    # Add pre-residual blocks
    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model_input)
    model = LeakyReLU(alpha=0.2)(model)

    model = self._dis_block(model, 64, 3, 2)
    model = self._dis_block(model, 128, 3, 1)
    model = self._dis_block(model, 128, 3, 2)
    model = self._dis_block(model, 256, 3, 1)
    model = self._dis_block(model, 256, 3, 2)
    model = self._dis_block(model, 512, 3, 1)
    model = self._dis_block(model, 512, 3, 2)

    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dense(1, activation='sigmoid')(model)

    # Create final model
    f_model = Model(inputs=model_input, outputs=model)
    return f_model
